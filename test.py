from circle_fit import least_squares_circle, plot_data_circle, hyper_fit
import matplotlib.pyplot as plt
from virtual_track import get_point_triple

points = [
    # (14.025877122223584, 2.611559243579859), (15.79200151437296, 3.080710137829354),
    # (16.7566533339147, 3.3775763401428436), (17.798138389736756, 3.9227231684357466),
    # (18.916779971546568, 4.348353766157228), (19.958472671413155, 4.781227043883984),
    # (20.80488357473378, 5.180942448510821), (21.913668930897778, 5.674186560626728),
    # (22.764712240152924, 6.099737185548592), (23.654217044570338, 6.617854529713524),
    # (24.5952292639615, 6.894839847338975), (25.500985306061295, 7.468515571383696),
    # (26.20384067826064, 7.967699883761224), (27.049258868401793, 8.496496435770274),
    # (27.727396899578544, 9.006888726673791), (28.43353610822672, 9.510196629412665),
    # (29.377799445281326, 10.182370065265033), (30.21536637327619, 11.176438875824088),
    # (31.28251908858862, 12.081966023028471), (32.1145568958177, 13.019821216770177),
    (32.592510461791, 13.901588414953327), (32.899348837452344, 14.891268716422141),
    (33.062412551166254, 16.001071504734774), (32.63719064531678, 17.064881813869317),
    (32.23126019316086, 18.47462998432196), (30.92749120772452, 20.009151412093658),
    (29.941800992771633, 20.61711175506125), (28.146038607469432, 20.99469266884644),
    (26.985927226317003, 20.93525621795775), (26.065552325277253, 20.852520738064346),
    # (25.10109580003577, 20.650846749950613), (24.376319420336763, 20.15724710905929),
    # (23.60561122690435, 19.888188460926102), (22.79636863109691, 19.275600941454165),
    # (22.041423359341824, 18.863250298102827), (21.34041822220923, 18.125473431729787),
    # (20.526728237828515, 17.470521076937825), (19.813556054629725, 16.80825718039783),
    # (18.466430524051933, 16.02865400348499), (17.357514477242038, 15.440527139436721),
    # (16.244081001088528, 14.860398892507266), (15.400489792888958, 14.378579195666987),
    # (14.479024944585095, 13.95672903541833), (13.095501694902863, 13.39008950149347),
    # (12.165577260810526, 12.986969618628429), (11.168246600407981, 12.622446453205308),
    # (10.116153677098225, 12.17493107497441), (8.668352824259525, 11.997480549191318),
    # (7.619860694477185, 11.892651025610999), (6.5241197861272315, 11.849812885672808),
    # (5.4751575660433165, 11.745270900224968), (4.3131381764810826, 11.819140606761016),
    # (2.986961573844729, 12.020659528421469), (1.9169579621207642, 12.338815743832757)
]

x = [p[0] for p in points]
y = [p[1] for p in points]

point_triple = get_point_triple(x, y, 1)

left_x = [p[0][0] for p in point_triple]
left_y = [p[0][1] for p in point_triple]


points += list(zip(left_x, left_y))

xc, yc, R, residu = hyper_fit(points)

# plt.plot([p[0] for p in points], [p[1] for p in points], '.')

plot_data_circle([p[0] for p in points], [p[1] for p in points], xc, yc, R)
plt.show()