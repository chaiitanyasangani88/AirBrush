# AirBrush

Airbrush is a tool built with opencv, to draw with every day objects. The tracker used is a combination of [CSRT Tracker](https://docs.opencv.org/3.4/d2/da2/classcv_1_1TrackerCSRT.html) and a colour mask. 

## Usage
1. Download/Clone the repo `git clone https://github.com/chaiitanyasangani88/AirBrush.git`
2. Create a virtual environment and source it.
`virtualenv venv`
`source venv/bin/activate`
3. Install dependencies: `pip3 install -r requirements.txt`
4. Run the file: `python new_tracker.py`

## Application usage
Keyboard Input
`s`: Select the region of interest
`t`: Toggle between write=True and write=False
`c`: Clear all the drawn items
`q`: quit

colours:
`r`: red
`g`: green
`b`: blue
