#!/home/imlab/.miniconda3/envs/carla/bin/python
from geodesy.utm import fromLatLong as proj
import numpy as np
import rospy
from traitlets.traitlets import Int
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import NavSatFix
from path_generator import path_generator
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA, Int64MultiArray
#from path_generator import gps_to_utm
#PG = path_generator()
waypoints = []
center = proj(35.64838, 128.40105, 0)
center = np.array([center.easting, center.northing, center.altitude])

turn_points_straight=np.round(np.array([
# [101,-70.7892944049,-80.3350438299,7,37,0], # straight
# [101, -5.45431420248,-152.636044776,42,2,0], # straight
# [102,-89.7552853322,-59.4080463387,43,5,0], # straight
# [102,-154.920326046,12.551952193, 3,45, 0],# straight
[200,-240.094256176,115.817951903, 10,9, 0], # short straight
[201,-158.496259121,25.5659516272, 8,8, 0], # short straight
[202,-157.729298761,21.8149483483, 8,8, 0], # short straight
[203,-74.6832844224,-66.9710493344, 8,8, 0], # short straight
[204,-74.0073031238,-70.6690488034, 8,8, 0] # short straight
]), 5)

turn_points=np.round(np.array([
[1,-263.219293277,84.6479525515,18,18,2],
[2,-245.165306888,108.270947973,7,4,2],
[3,-233.082265306,119.649953682,13,1,2],
[4,-218.476276029,133.752956838,11,9,2],
[5,-188.868884425,0.924987147097,7,5,2],
[6,-166.544295595,22.1079528229,12,8,2],
[7,-160.912310905,27.6009516199,11,8,1],
[8,-154.834389636,31.5651654066,9,5,2],
[9,-138.608199347,45.6767437309,7,7,2],
[10,-183.284975067,-3.06565001933,13,6,2],
[11,-162.779261785,19.596954837,5,9,1],
[12,-161.588029117,15.8534738831,7,8,2],
[13,-156.179315607,19.8749486036,7,8,1],
[14,-153.21528748,27.3359509846,6,10,1],
[15,-149.194295593,26.7539528208,10,5,2],
[16,-134.176255844,40.8333998751,8,4,2],
[17,-101.270215314,-88.2169306972,8,6,2],
[18,-83.1722695882,-69.7770473845,10,7,2],
[19,-78.4022887377,-63.7600527359,8,6,1],
[20,-70.927552853, -60.6896212869,8,7,2],
[21,-54.8138473506,-47.3321649865,9,4,2],
[22,-95.907697227,-92.6747719357,13,6,2],
[23,-78.2483199305,-69.0910516484,6,7,1],
[24,-78.4266286552,-76.726239901,5,5,2],
[25,-73.9282955965,-73.3310471787,9,7,1],
[26,-69.8442955929,-69.3200471788,8,5,1],
[27,-65.8123092938,-66.8740458237,6,5,2],
[28,-50.6066547912,-51.605281835,7,5,2],
[29,-2.38426054391,-163.527044795,18,30,2],
[30,11.979722245,-164.778048522,8,10,1],
[31,15.972704403,-163.400047178,6,6,2],
[32,17.7477044033,-161.900047179,8,6,2],
[33,37.950712637,-149.167057891,7,5,2]
]), 5)

''' idx: 102 '''
straight_1_points=np.round(np.array([[-156.931284799764,14.7749488279223,1.12359550561798], [-154.920326046471,12.5519521930255,2.24719101123596], [-152.909281853179,10.3279572888277,3.37078651685393], [-152.852270206669,10.2649535052478,4.49438202247191], [-150.835322189028,8.04395098192617,5.61797752808989], [-148.818276462029,5.82395766396076,6.74157303370787], [-148.350305880711,5.30795396631584,7.86516853932584], [-146.32625142223,3.09395215800032,8.98876404494382], [-145.017321199877,1.66195023199543,10.1123595505618], [-143.009290150541,-0.567048317287117,11.2359550561798], [-141.000256121915,-2.79504319606349,12.3595505617978], [-140.007259570935,-3.89805057924241,13.4831460674157], [-137.994312912924,-6.1230517742224,14.6067415730337], [-135.982270519831,-8.34804696403444,15.7303370786517], [-133.969315741968,-10.5720490198582,16.8539325842697], [-132.440252624394,-12.2630472402088,17.9775280898876], [-131.219334297057,-13.6390523733571,19.1011235955056], [-129.197281194211,-15.8560436209664,20.2247191011236], [-127.176313548407,-18.0730499364436,21.3483146067416], [-126.059306522948,-19.2980422414839,22.4719101123595], [-124.047264302324,-21.5240439013578,23.5955056179775], [-122.036307470233,-23.750049544964,24.7191011235955], [-120.025259029761,-25.9760441784747,25.8426966292135], [-118.014300175302,-28.2020488725975,26.9662921348315], [-116.002253906336,-30.428048635833,28.0898876404494], [-115.591303860885,-30.8830433823168,29.2134831460674], [-113.57625053369,-33.1060440610163,30.3370786516854], [-112.226284868259,-34.5950462361798,31.4606741573034], [-110.214326220972,-36.8210440841503,32.5842696629214], [-109.208262150001,-37.9350526630878,33.7078651685393], [-107.192294171953,-40.1570480819792,34.8314606741573], [-106.542310187651,-40.8740501762368,35.9550561797753], [-104.53025815502,-43.1000472377054,37.0786516853933], [-102.551294882956,-45.289051596541,38.2022471910112], [-100.535323557793,-47.5110454410315,39.3258426966292], [-98.5202626688406,-49.7340425648727,40.4494382022472], [-96.5052852662629,-51.9560515633784,41.5730337078652], [-95.6492772161728,-52.900042864494,42.6966292134831], [-93.635306435579,-55.1240525795147,43.8202247191011], [-91.6213284184341,-57.3470414471813,44.9438202247191], [-89.755285332154,-59.4080463387072,46.0674157303371], [-88.9223268347559,-60.3760423143394,47.1910112359551], [-86.9073201687425,-62.5940451947972,48.314606741573], [-84.8933022167184,-64.8110433295369,49.438202247191], [-82.8782935347408,-67.0290452572517,50.561797752809], [-80.8632838409394,-69.2470467090607,51.685393258427], [-78.8482670519734,-71.464049495291,52.8089887640449], [-76.8342511527007,-73.6820439202711,53.9325842696629], [-74.8193289634073,-75.9000433944166,55.0561797752809], [-72.8043091426953,-78.117044754792,56.1797752808989], [-70.789294404909,-80.3350438298657,57.3033707865169], [-68.9672667441191,-82.3410505387001,58.4269662921348], [-66.9652998466045,-84.5760498894379,59.5505617977528], [-64.9633258470567,-86.8100505792536,60.6741573033708], [-62.961260311713,-89.0440513538197,61.7977528089888], [-60.9592842936399,-91.2780511057936,62.9213483146067], [-60.7313351100311,-91.5320485206321,64.0449438202247], [-58.716254275234,-93.7550473422743,65.1685393258427], [-56.7012568662758,-95.9770469446667,66.2921348314607], [-54.6862645383226,-98.2000442631543,67.4157303370787], [-54.2542518628179,-98.6760496161878,68.5393258426966], [-52.2552773108473,-100.91304585943,69.6629213483146], [-51.2673406815738,-102.018041656818,70.7865168539326], [-49.2433342386503,-104.233051366173,71.9101123595506], [-48.5702865777421,-104.970043274108,73.0337078651685], [-47.5782559405197,-106.057051140815,74.1573033707865], [-45.5692646243842,-108.285045190249,75.2808988764045], [-43.5602723647025,-110.513049858157,76.4044943820225], [-42.588277619856,-111.590052517131,77.5280898876405], [-40.5782879749895,-113.81805146439,78.6516853932584], [-38.5692870529019,-116.045045677572,79.7752808988764], [-37.1742870605667,-117.59104300011,80.8988764044944], [-35.1572811043588,-119.812051470857,82.0224719101124], [-34.1552955493098,-120.915045722853,83.1460674157303], [-32.1442997642443,-123.142050564289,84.2696629213483], [-31.6923327522818,-123.642045133747,85.3932584269663], [-29.6793317880947,-125.867042972706,86.5168539325843], [-27.666323725367,-128.091042145621,87.6404494382023], [-26.0183349713334,-129.913047952577,88.7640449438202], [-24.0023253139807,-132.135045908857,89.8876404494382], [-21.9863146448624,-134.357043389697,91.0112359550562], [-20.3343340590363,-136.178052002564,92.1348314606742], [-18.3203254218679,-138.402043951675,93.2584269662921], [-16.3063097550767,-140.625048325397,94.3820224719101], [-14.9362689498812,-142.138048174325,95.5056179775281], [-12.9292652414297,-144.368052061647,96.6292134831461], [-10.9222604566603,-146.598044383805,97.7528089887641], [-9.4872625269345,-148.193049818277,98.876404494382], [-7.47133508784464,-150.415043319576,100]]), 5)
''' idx: 103 '''
straight_right_points=np.round(np.array([[-212.757304455736,130.3689498147,1.23456790123457], [-210.852260651649,128.256958086044,2.46913580246914], [-208.941283869615,126.137953990605,3.7037037037037], [-207.208253519435,124.216955869459,4.93827160493827], [-205.262297439796,122.058951949701,6.17283950617284], [-203.189284095715,119.759953620844,7.40740740740741], [-201.369302153529,117.741951701231,8.64197530864197], [-199.54625972244,115.720947865397,9.87654320987654], [-197.932279587607,113.93195780227,11.1111111111111], [-196.499312521832,112.342955296859,12.3456790123457], [-194.194268448395,109.791951733641,13.5802469135802], [-191.802267578023,107.144953206647,14.8148148148148], [-189.206325163424,104.271951779258,16.0493827160494], [-186.329279589409,101.088957454078,17.283950617284], [-183.661308226001,98.1359474766068,18.5185185185185], [-180.707284551812,94.8679567105137,19.7530864197531], [-177.762262895762,91.6079523772933,20.9876543209876], [-174.410250348912,87.8979556271806,22.2222222222222], [-171.046273973538,84.1749544856139,23.4567901234568], [-168.022285290004,80.8289560200647,24.6913580246914], [-165.210334549192,77.7169525725767,25.9259259259259], [-161.872336482746,74.0229515610263,27.1604938271605], [-158.005285053165,69.7349565033801,28.3950617283951], [-154.673262122436,66.0399522618391,29.6296296296296], [-151.087307686394,62.0619529825635,30.8641975308642], [-147.803304519679,58.4189530676231,32.0987654320988], [-144.516317286238,54.7749490649439,33.3333333333333], [-141.178266421251,51.07194969384,34.5679012345679], [-137.73625582899,47.2549511916004,35.8024691358025], [-134.132264491695,43.2779580964707,37.037037037037], [-130.655328612251,39.4419492073357,38.2716049382716], [-127.245298617519,35.6789548038505,39.5061728395062], [-124.409251433273,32.5499477596022,40.7407407407407], [-121.679295615584,29.5409553973004,41.9753086419753], [-118.47928464273,26.0159490616061,43.2098765432099], [-115.415328256204,22.6409542760812,44.4444444444444], [-112.349311781407,19.2619551178068,45.679012345679], [-109.418264041655,16.0329495966434,46.9135802469136], [-105.891276789305,12.1469557848759,48.1481481481482], [-102.550253093184,8.46594785479829,49.3827160493827], [-99.4792720195837,5.08295234246179,50.6172839506173], [-96.7742899297737,2.1029581814073,51.8518518518519], [-94.3032992241206,-0.620049555320293,53.0864197530864], [-90.7722970794421,-4.50304467789829,54.320987654321], [-87.0983055550023,-8.54304484743625,55.5555555555556], [-83.2163095564465,-12.8120417245664,56.7901234567901], [-78.8552659042762,-17.6080515650101,58.0246913580247], [-74.3833367628977,-22.5260425340384,59.2592592592593], [-70.5033141637105,-26.794045469258,60.4938271604938], [-67.4742696039611,-30.1250431030057,61.7283950617284], [-65.1613231074298,-32.6690498013049,62.962962962963], [-62.0242844320019,-36.1210461454466,64.1975308641975], [-58.8832782728714,-39.5760491108522,65.4320987654321], [-55.5252810675302,-43.271050570067,66.6666666666667], [-52.5242942082696,-46.5730451098643,67.9012345679012], [-49.2153021093691,-50.2140516340733,69.1358024691358], [-46.0582871835795,-53.6880472241901,70.3703703703704], [-42.7453246461228,-57.3320479206741,71.6049382716049], [-39.688305152813,-60.6960478373803,72.8395061728395], [-37.1123057637597,-63.5300497640856,74.0740740740741], [-34.7003370739403,-66.1840464868583,75.3086419753087], [-31.8403261798085,-69.3430487634614,76.5432098765432], [-28.6823088360252,-72.8330488759093,77.7777777777778], [-25.0912921755807,-76.8010431244038,79.0123456790123], [-21.7072747129132,-80.5390469580889,80.2469135802469], [-18.3263202616945,-84.2750454009511,81.4814814814815], [-15.549304073269,-87.3430525404401,82.716049382716], [-13.2613014483359,-89.8700451231562,83.9506172839506], [-10.7332888644887,-92.6630521439947,85.1851851851852], [-6.87528778950218,-96.9150419682264,86.4197530864197], [-3.20227060251636,-100.964050236158,87.6543209876543], [0.946743694657926,-105.537042764947,88.8888888888889], [5.20970483636484,-110.235051100608,90.1234567901235], [9.7116968575865,-115.197047156282,91.358024691358], [13.9967015569564,-119.919042963069,92.5925925925926], [17.9246852677898,-124.249043249059,93.8271604938272], [21.6497117046965,-128.354047453031,95.0617283950617], [25.5886808402138,-132.69504720578,96.2962962962963], [28.8537286421051,-136.29304244183,97.5308641975309], [32.484689341567,-140.29604916228,98.7654320987654], [35.7857121576089,-143.934045666829,100]]), 5)

target_update_pub = rospy.Publisher("/map/target_update", MarkerArray, queue_size=10)
turn_update_pub = rospy.Publisher("/map/target_turnpoints", Int64MultiArray, queue_size=10)

def get_target_waypoints(waypoints, current_x, current_y, n):
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        distance = np.sqrt((waypoints[i][0] - current_x) ** 2 + (waypoints[i][1] - current_y) ** 2)
        if distance <= closest_len:
            closest_len = distance
            closest_index = i
    end_index = np.min([closest_index + n, len(waypoints)])

    return waypoints[closest_index:end_index], gene_cmd(waypoints)[0][closest_index:end_index]

def gps_to_utm(latitude, longitude, altitude):
    pos = proj(latitude, longitude, altitude)
    pos = np.array([pos.easting, pos.northing, pos.altitude])
    pos[:2] -= center[:2]
    return pos

def gene_cmd(waypoints):
        wapoints_cmd = np.zeros(len(waypoints))
        wapoints_idx = np.zeros(len(waypoints))
        wapoints_per = np.zeros(len(waypoints))

        arr_waypoints = np.round(np.array(waypoints), 5)

        ''' long straight '''
        for i in range(len(arr_waypoints)):
            if arr_waypoints[i][0].tolist() in straight_1_points[:, 0].tolist() and arr_waypoints[i][1].tolist() in straight_1_points[:, 1].tolist():
                tmp_idx = straight_1_points[:, 0].tolist().index(arr_waypoints[i][0])
                wapoints_idx[i] = 102
                wapoints_per[i] = straight_1_points[tmp_idx, 2]
                # print('tmp_idx :', tmp_idx,' ' , arr_waypoints[i][0], ' ',straight_1_points[tmp_idx, 0], ' ',straight_1_points[tmp_idx, 2])
        for i in range(len(arr_waypoints)):
            if arr_waypoints[i][0].tolist() in straight_right_points[:, 0].tolist() and arr_waypoints[i][1].tolist() in straight_right_points[:, 1].tolist():
                tmp_idx = straight_right_points[:, 0].tolist().index(arr_waypoints[i][0])
                wapoints_idx[i] = 103
                wapoints_per[i] = straight_right_points[tmp_idx, 2]


        ''' short straight '''
        for i in range(len(arr_waypoints)):
            for j in range(len(turn_points_straight)):
                if arr_waypoints[i][0] == turn_points_straight[j][1] and arr_waypoints[i][1] == turn_points_straight[j][2]:
                    cmd = turn_points_straight[j][5]
                    min_idx = np.max([0, i - int(turn_points_straight[j][3])])
                    max_idx = np.min([len(arr_waypoints)-1, i + int(turn_points_straight[j][4])])
                    wapoints_cmd[min_idx:max_idx] = cmd
                    wapoints_idx[min_idx:max_idx] = turn_points_straight[j][0]
                    for n in range(min_idx, max_idx):
                        wapoints_per[n] = (n - i + int(turn_points_straight[j][3])) / (int(turn_points_straight[j][3]) + int(turn_points_straight[j][4]))

        ''' turn '''
        for i in range(len(arr_waypoints)):
            for j in range(len(turn_points)):
                if arr_waypoints[i][0] == turn_points[j][1] and arr_waypoints[i][1] == turn_points[j][2]:
                    cmd = turn_points[j][5]
                    min_idx = np.max([0, i - int(turn_points[j][3])])
                    max_idx = np.min([len(arr_waypoints)-1, i + int(turn_points[j][4])])
                    wapoints_cmd[min_idx:max_idx] = cmd
                    wapoints_idx[min_idx:max_idx] = turn_points[j][0]
                    for n in range(min_idx, max_idx):
                        wapoints_per[n] = (n - i + int(turn_points[j][3])) / (int(turn_points[j][3]) + int(turn_points[j][4]))

        return wapoints_cmd, wapoints_idx, wapoints_per

def callback_map_target(message):
    global waypoints
    waypoints = []
    for m in message.markers:
        waypoints.append([m.points[0].x, m.points[0].y, 8.333])

def callback(gps_sub):
    out_msg = MarkerArray()
    turn_msg = Int64MultiArray()
    point = gps_to_utm(gps_sub.latitude, gps_sub.longitude, 0)
    target_waypoints, target_turnpoints = get_target_waypoints(waypoints, point[0], point[1], 20)
    for target_w in target_waypoints:
        marker = Marker()
        marker.pose.position = Point(x=target_w[0], y=target_w[1], z=target_w[-1])
        marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        marker.type = Marker.SPHERE
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.scale = Vector3(x=1.5, y=1.5, z=1.5)
        #marker.id = i
        marker.color = ColorRGBA(0, 1, 1, 1)
        out_msg.markers.append(marker)

    turn_msg.data = list(target_turnpoints.astype(int))
    target_update_pub.publish(out_msg)
    turn_update_pub.publish(turn_msg)
def getRosFnc():
    rospy.init_node('getRosFnc_main', anonymous=True)
    rospy.Subscriber('/map/target', MarkerArray, callback_map_target)
    rospy.Subscriber('/fix', NavSatFix, callback)
    rospy.spin()

if __name__ == '__main__':
    getRosFnc()
