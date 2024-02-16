import sys
import cv2
from haversine import haversine, Unit
import gpxpy
import requests
import time
from math import ceil


gps_with_time = []
start_time = 0


def process_gps_file(file_path):
    """Process the input GPX file and extract GPS coordinates with timestamps."""
    with open(file_path, "r") as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        initial_time = None
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if initial_time is None:
                        initial_time = point.time
                        time_in_seconds = 0
                    else:
                        time_difference = point.time - initial_time
                        time_in_seconds = int(time_difference.total_seconds())
                    data = {
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "time_seconds": time_in_seconds,
                    }
                    gps_with_time.append(data)

    for i in range(1, len(gps_with_time)):
        time_difference = gps_with_time[i]['time_seconds'] - gps_with_time[i - 1]['time_seconds']
        distance = haversine(
            (gps_with_time[i - 1]['lat'], gps_with_time[i - 1]['lon']),
            (gps_with_time[i]['lat'], gps_with_time[i]['lon']),
            unit=Unit.KILOMETERS
        )
        speed_kph = distance / time_difference * 3600 if time_difference != 0 else 0
        gps_with_time[i]['speed_kph'] = speed_kph


def make_video(cap, text, time_wait, el_start_time, out):
    """Generate a video with instructions and elapsed time overlay."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    curr_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            elapsed_time = current_time - el_start_time
            display_time = time.time() - curr_time
            if display_time <= time_wait:
                for i, line in enumerate(text):
                    y = 40 + i * 45
                    cv2.putText(frame, line, (15, y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                elapsed_time_str = f"Elapsed Time: {int(elapsed_time)}s"
                cv2.putText(frame, elapsed_time_str, (15, 40 + len(text) * 45), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                out.write(frame)
                cv2.imshow("Frame", frame)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


def final_output(text, time_wait, cap, el_start_time, out):
    """Generate the final output frame with instructions and elapsed time overlay."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    curr_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            elapsed_time = current_time - el_start_time
            display_time = time.time() - curr_time
            if display_time <= time_wait+5:
                cv2.putText(frame, text, (15, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                elapsed_time_str = f"Elapsed Time: {int(elapsed_time)}s"
                cv2.putText(frame, elapsed_time_str, (15, 80), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                out.write(frame)
                cv2.imshow("Frame", frame)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


def walk_command(instruction, time_takes, cap, elapsed_start_time, out):
    """Process walking instructions and generate corresponding video frames."""
    global start_time
    for i in range(len(gps_with_time)):
        walk = gps_with_time[i]
        current_walk_time = walk["time_seconds"]
        time_difference = 0
        if start_time <= current_walk_time <= (start_time + time_takes):
            if i > 0:
                time_difference = (current_walk_time - gps_with_time[i - 1]["time_seconds"])
            text = (
                f"{instruction}\n"
                f"Stay center on the same street\n"
                f"Location: lat={walk['lat']} , lon={walk['lon']}\n"
                f"Time Takes: {time_takes}s\n"
            )
            try:
                if walk['speed_kph'] > 0:
                    walking_speed = f"Walking Speed: {walk['speed_kph']:.2f} km/h"
                    text += walking_speed
                else:
                    walking_speed = f"Walking Speed: Not Walking km/h"
                    text += walking_speed
            except KeyError:
                pass
            lines = text.split("\n")
            make_video(cap, lines, time_difference, elapsed_start_time, out)
    start_time += time_takes


def commands(type_input, instructions, final_instruction, time_takes, cap, elapsed_start_time, out):
    """Process general instructions and call the appropriate function."""
    if type_input == 1:
        instruction = f"Walk: {instructions}"
        walk_command(instruction, time_takes, cap, elapsed_start_time, out)

    if type_input == 15:
        instruction = f"Go left: {instructions}"
        walk_command(instruction, time_takes, cap, elapsed_start_time, out)

    if type_input == 10:
        instruction = f"Go right: {instructions}"
        walk_command(instruction, time_takes, cap, elapsed_start_time, out)

    if type_input == 8:
        instruction = f"Stay center: {instructions}"
        walk_command(instruction, time_takes, cap, elapsed_start_time, out)

    if type_input == 4:
        instruction = f"Stay center: {final_instruction}"
        final_output(instruction, time_takes, cap, elapsed_start_time, out)


def travel_in_api(trace_data, cap, elapsed_start_time, out):
    """Iterate through the API trace data and call commands"""
    for trace_point in trace_data["trip"]["legs"][0]["maneuvers"]:
        try:
            commands(
                trace_point["type"],
                trace_point["verbal_post_transition_instruction"],
                trace_point["verbal_pre_transition_instruction"],
                ceil(trace_point["time"]),
                cap,
                elapsed_start_time,
                out
            )
        except Exception as e:
            commands(
                trace_point["type"],
                trace_point["verbal_pre_transition_instruction"],
                trace_point["verbal_pre_transition_instruction"],
                ceil(trace_point["time"]),
                cap,
                elapsed_start_time,
                out
            )


def main():
    """Main function to start the video generation process."""
    if len(sys.argv) != 4:
        print("Usage: python generate_video_walk.py <input_video_path> <output_video_path> <gps_coordinates_path>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    gps_coordinates_path = sys.argv[3]

    process_gps_file(gps_coordinates_path)

    payload = {
        "id": "group-4",
        "shape": [{"lat": data["lat"], "lon": data["lon"]} for data in gps_with_time],
        "costing": "pedestrian",
        "language": "en",
        "units": "kilometers",
        "shape_match": "map_snap",
    }

    valhalla_base_url = "https://valhalla1.openstreetmap.de"
    response = requests.post(f"{valhalla_base_url}/trace_route", json=payload)
    trace_data = response.json()

    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path+'output.avi', cv2.VideoWriter_fourcc("M", "J", "P", "G"), 24, (frame_width, frame_height))

    elapsed_start_time = time.time()
    print('Video is in Process......')
    travel_in_api(trace_data, cap, elapsed_start_time, out)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('Video Processed Successfully !!')


if __name__ == "__main__":
    main()
