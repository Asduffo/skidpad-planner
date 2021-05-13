# Skidpad-Planner

This is a planner-code for the competition category "Skidpad" in [Formula Student](https://www.formulastudent.de/fsg/).
In this category, the car has to drive through a parkour of two circles, going through 
both of them 2 times.

This planner gets his path by fitting circles on to the coordinates of cones, using the
least-squares method of [circle-fit](https://pypi.org/project/circle-fit/).

## Run code an example-track

1. Clone this repository
```shell
git clone https://github.com/nathanieltornow/skidpad-planner.git
cd skidpad-planner
```

2. Install the dependencies
```shell
pip install -r requirements.txt
```

3. Run the code
```shell
python main.py
```

