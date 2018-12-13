from objdetection.msg import Object, Fusion
import numpy as np
import rospy

dist = []
ang = []
def callback(data):
    if len(data.distance) > 0:
        dist.append(data.distance[0])
        ang.append(data.angle[0])
    if len(dist) == 500:
        n_dist = np.asarray(dist)
        n_ang = np.asarray(ang)
        # get evaluation values
        print np.mean(n_dist), np.var(n_dist), np.std(n_dist), np.mean(n_ang), np.var(n_ang), np.std(n_ang)

def listener():
    num = 0
    rospy.init_node('custom_listener', anonymous=True)
    rospy.Subscriber('/fusion/human',Fusion, callback)
    rospy.spin()

if __name__== '__main__':
    listener()
    print('hi')
