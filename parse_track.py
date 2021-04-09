import matplotlib.pyplot as plt
import yaml


def parse_track(track_name):
    with open(track_name) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
        res = []
        i = 0
        for c in content['cones_left']:
            res.append(c)
            i += 1
        for c in content['cones_right']:
            res.append(tuple(c))
            i += 1
        x = content['starting_pose'][0]
        y = content['starting_pose'][1]
        direction = content['starting_pose'][2]

        return res, x, y, direction

#
# r, x, y, _ = parse_track('skidpad.yaml')
# plt.plot([p[0] for p in r], [p[1] for p in r], '.')
# plt.plot(x, y, 'o')
# plt.axis('equal')
# plt.show()


