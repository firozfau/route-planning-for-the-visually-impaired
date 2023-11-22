# route-planning-for-the-visually-impaired

In the initiative entitled "Road Scene Understanding for the Visually Impaired", our research team is meticulously advancing the development of the Sidewalk Environment Detection System for Assistive NavigaTION (hereinafter referred to as SENSATION). The primary objective of this venture is to enhance the mobility capabilities of blind or visually impaired persons (BVIPs) by ensuring safer and more efficient navigation on pedestrian pathways.
For the implementation phase, a specialized apparatus has been engineered: a chest-mounted bag equipped with an NVIDIA Jetson Nano, serving as the core computational unit. This device integrates a plethora of sensors including, but not limited to, tactile feedback mechanisms (vibration motors) for direction indication, optical sensors (webcam) for environmental data acquisition, wireless communication modules (Wi-Fi antenna) for internet connectivity, and geospatial positioning units (GPS sensors) for real-time location tracking.
Despite the promising preliminary design of the prototype, several technical challenges persist that warrant investigation.
The "Road Scene Understanding for the Visually Impaired" initiative is actively seeking student collaborators to refine the Jetson Nano-fueled SENSATION system. Through the combination of GPS systems and cutting-edge image segmentation techniques refined for sidewalk recognition, participating teams are expected to architect an application tailored to aid BVIPs in traversing urban landscapes, seamlessly guiding them from a designated starting point to a predetermined destination.
The developmental framework for this endeavor is based on the Python programming language; hence, prior experience with Python will be an advantage.
For the purposes of real-world testing and calibration, the navigation part will start at the main train station in Erlangen and end at the University Library of Erlangen-Nuremberg (Schuhstrasse 1a).
Technical milestones that must be completed in this project include:
Algorithmic generation of navigational pathways in Python, depending on defined start and endpoint parameters.
Real-time geospatial tracking to determine the immediate coordinates of the BVIP.
Optical recording of the current coordinates and subsequent algorithmic evaluation to check the orientation of the sidewalk.
Useful links for this project:
1. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, https://link.springer.com/chapter/10.1007/978-3-030-01234-2_49
2. How to connect an USB GPS receiver with a Linux computer?, https://gpswebshop.com/blogs/tech-support-by-os-linux/how-to-connect-an-usb-gps-receiver-with-a-linux-computer
3. Open Streetmap, https://www.openstreetmap.de/
4. GeoPy documentation, https://geopy.readthedocs.io/en/stable/
5. Efficient Localisation Using Images and OpenStreetMaps, http://www.ipb.uni-bonn.de/pdfs/zhou2021iros.pdf
6. A Practical Guide to an Open-Source Map-Matching Approach for Big GPS Data, https://link.springer.com/article/10.1007/s42979-022-01340-5
7. VALHALLA Map Matching service API reference, https://valhalla.github.io/valhalla/api/map-matching/api-reference/
8. L2MM: Learning to Map Matching with Deep Models for Low-Quality GPS Trajectory Data, https://dl.acm.org/doi/10.1145/3550486

------------------------------------------------------------------------------------------
<!-- GETTING STARTED -->
## Getting Started

To obtain information from Erlangen Hauptbahnhof to the University library located on the pedestrian walking street using the VALHALLA MAP API, please follow the correct syntax for your API requests. Ensure you include accurate details such as the starting point (Erlangen Hauptbahnhof), destination (University Library), and any necessary parameters specified by the VALHALLA MAP API documentation.

Dependency: 
* Python
  ```sh
  1. Visit python website and download  https://www.python.org/downloads/ 
  2. Valhalla REST API: https://valhalla.github.io/valhalla/api/map-matching/api-reference/
  3. Google map (For obtaining latitude and longitude as API parameters.)
  ```

Environment: 
* route-planning-for-the-visually-impaired
  ```sh
  1. visit to GitLab location : https://gitlab.rrze.fau.de/rsu-vi/route-planning-for-the-visually-impaired
  2. Find and select your profile and visit SSH keys tab
  3. Add your public ssh key here for authorize access.
  3. Then again go to project location "route-planning-for-the-visually-impaired"
  4. copy SSH url: git@gitlab.rrze.fau.de:rsu-vi/route-planning-for-the-visually-impaired.git
  5. Go to you local drive and select project location then open terminal and clone this SSH URL.
  ```
Project management: 
* Clone the project 
  ```sh
  1. git clone  SSH-URL
  2. git checkout group-4
  3. git pull
  ```
* Manage project 
  ```sh
  1. git pull
  2. git commit -m "write which purpose you update and add your source code"
  3. git push origin group-4
  ```

* Project file description 
 `.env`
  ```sh
  This file is used to store all global variables that can be accessed from anywhere within this project.
  ```
 `main.py`
  ```sh
  The 'main.py' file serves as the primary file for this project, encompassing all dependency files and initializing the necessary variables, generating API links, and more.
  ```
 `APIDataProcessor.py`
  ```sh
  In the 'APIDataProcessor.py' file, here all API realted work
  ```

 `BVIPSimulator.py`
   ```js
    This file containt all BVIP simulator related task
   ```     



 `How to check Last Issues`
   ```js
     '*' Open terminal then go to the roor directory and command: python main.py 
   ```    
