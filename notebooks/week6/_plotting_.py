import numpy as np
import matplotlib.pyplot as plt

np.random.seed(22)

x_c1 = 1.5 * np.random.random(4)
y_c1 = 4 * np.random.random(4)

x_c2 = 2 + 1.5 * np.random.random(5)
y_c2 = 4 * np.random.random(5)




def _plot_trees_():
    import math
    
    centers = [[0.5, 0.6], [0.3, 0.2], [0.7, 0.2]]
    radius = 0.1 

    tan_1 = (centers[1][1] - centers[0][1])/(centers[1][0] - centers[0][0])
    tan_2 = (centers[2][1] - centers[0][1])/(centers[2][0] - centers[0][0])

    cos_1 = np.sqrt((1+tan_1**2)**(-1))
    cos_2 = np.sqrt((1+tan_2**2)**(-1))

    sin_1 = 1 - cos_1**2
    sin_2 = 1 - cos_2**2

    offset_points_c1 = [0.1*np.log10(x_c1)-0.3*0.1, 0.1*np.log10(y_c1)]
    offset_points_c2 = [0.2*np.log10(x_c2)-0.03, 0.05*np.log10(y_c2)+0.02]

    circle1 = plt.Circle(centers[0], radius, fill=False, color="k")
    circle2 = plt.Circle(centers[1], radius, fill=False, color="k")
    circle3 = plt.Circle(centers[2], radius, fill=False, color="k")

    fig, ax = plt.subplots()

    plt.plot(centers[0][0]+offset_points_c1[0], 
             centers[0][1]+offset_points_c1[1], "r.", markersize=12)

    plt.plot(centers[0][0]+offset_points_c2[0], 
             centers[0][1]+offset_points_c2[1], "b.", markersize=12)



    plt.plot(centers[1][0]+offset_points_c1[0], 
             centers[1][1]+offset_points_c1[1], "r.", markersize=12)

    plt.plot(centers[2][0]+offset_points_c2[0], 
             centers[2][1]+offset_points_c2[1], "b.", markersize=12)


    plt.plot([centers[1][0]+1.1*radius*cos_1, centers[0][0]-1.1*radius*cos_1], 
             [centers[1][1]+1.1*radius*sin_1, centers[0][1]-1.1*radius*sin_1], color="k")

    plt.plot([centers[2][0]-1.1*radius*cos_2, centers[0][0]+1.1*radius*cos_2], 
             [centers[2][1]+1.1*radius*sin_2, centers[0][1]-1.1*radius*sin_2], color="k")


    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

    plt.text(centers[1][0]-0.035, centers[1][1]+0.25, r"$x < 1.8$")
    plt.text(centers[2][0]-0.1, centers[2][1]+0.25, r"$x \geq 1.8$")


    plt.text(centers[1][0]-0.035, centers[1][1]-0.15, r"Red", color="r")
    plt.text(centers[2][0]-0.035, centers[2][1]-0.15, r"Blue", color="b")

    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.8)

    plt.axis("off")
    
    return fig, ax





def _plot_tree_2():
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(22)

    x_c1 = 1.5 * np.random.random(4)
    y_c1 = 4 * np.random.random(4)

    x_c2 = 2 + 1.5 * np.random.random(5)
    y_c2 = 4 * np.random.random(5)

    x_c3 = -0.3 + 2.2*np.random.random(4)
    y_c3 = 3.5 + np.random.random(4)


    import math

    centers = [[0.5, 0.6], [0.3, 0.2], [0.7, 0.2], [0.1, -0.2], [0.5, -0.2]]
    radius = 0.1 

    tan_1 = (centers[1][1] - centers[0][1])/(centers[1][0] - centers[0][0])
    tan_2 = (centers[2][1] - centers[0][1])/(centers[2][0] - centers[0][0])

    cos_1 = np.sqrt((1+tan_1**2)**(-1))
    cos_2 = np.sqrt((1+tan_2**2)**(-1))

    sin_1 = 1 - cos_1**2
    sin_2 = 1 - cos_2**2

    offset_points_c1 = [0.1*np.log10(x_c1)-0.3*0.1, 0.1*np.log10(y_c1)]
    offset_points_c2 = [0.4*np.log10(x_c2)-0.13, 0.05*np.log10(y_c2)+0.008]
    offset_points_c3 = [0.09*x_c3-0.08, 0.1*y_c3-0.338]

    circle1 = plt.Circle(centers[0], radius, fill=False, color="k")
    circle2 = plt.Circle(centers[1], radius, fill=False, color="k")
    circle3 = plt.Circle(centers[2], radius, fill=False, color="k")
    circle4 = plt.Circle(centers[3], radius, fill=False, color="k")
    circle5 = plt.Circle(centers[4], radius, fill=False, color="k")

    fig, ax = plt.subplots()

    size_ = 9.5

    plt.plot(centers[0][0]+offset_points_c1[0], 
             centers[0][1]+offset_points_c1[1], "r.", markersize=size_)

    plt.plot(centers[0][0]+offset_points_c2[0], 
             centers[0][1]+offset_points_c2[1], "b.", markersize=size_)

    plt.plot(centers[0][0]+offset_points_c3[0], 
             centers[0][1]+offset_points_c3[1], "g.", markersize=size_)


    # circle 2
    plt.plot(centers[1][0]+offset_points_c3[0], 
             centers[1][1]+offset_points_c3[1], "g.", markersize=size_)



    # circle 3
    plt.plot(centers[1][0]+offset_points_c1[0], 
             centers[1][1]+offset_points_c1[1], "r.", markersize=size_)

    plt.plot(centers[2][0]+offset_points_c2[0], 
             centers[2][1]+offset_points_c2[1], "b.", markersize=size_)


    # circle 4
    plt.plot(centers[3][0]+offset_points_c1[0], 
             centers[3][1]+offset_points_c1[1], "r.", markersize=size_)

    # circle 5
    plt.plot(centers[4][0]+offset_points_c3[0], 
             centers[4][1]+offset_points_c3[1], "g.", markersize=size_)




    plt.plot([centers[1][0]+1.1*radius*cos_1, centers[0][0]-1.1*radius*cos_1], 
             [centers[1][1]+1.1*radius*sin_1, centers[0][1]-1.1*radius*sin_1], color="k")

    plt.plot([centers[2][0]-1.1*radius*cos_2, centers[0][0]+1.1*radius*cos_2], 
             [centers[2][1]+1.1*radius*sin_2, centers[0][1]-1.1*radius*sin_2], color="k")



    plt.plot([centers[1][0]-1.1*radius*cos_1, centers[3][0]+1.1*radius*cos_1], 
             [centers[1][1]-1.1*radius*sin_1, centers[3][1]+1.1*radius*sin_1], color="k")

    plt.plot([centers[1][0]+1.1*radius*cos_2, centers[4][0]-1.1*radius*cos_2], 
             [centers[1][1]-1.1*radius*sin_2, centers[4][1]+1.1*radius*sin_2], color="k")




    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    ax.add_artist(circle5)





    plt.text(centers[1][0]-0.035, centers[1][1]+0.25, r"$x < 1.8$", fontsize=14)
    plt.text(centers[2][0]-0.1, centers[2][1]+0.25, r"$x \geq 1.8$", fontsize=14)

    plt.text(centers[3][0]-0.035, centers[3][1]+0.25, r"$y < 3.0$", fontsize=14)
    plt.text(centers[4][0]-0.1, centers[4][1]+0.25, r"$y \geq 3.0$", fontsize=14)

    plt.text(centers[2][0]-0.035, centers[2][1]-0.15, r"Blue", color="b")
    plt.text(centers[3][0]-0.035, centers[3][1]-0.15, r"Red", color="r")
    plt.text(centers[4][0]-0.055, centers[4][1]-0.15, r"Green", color="g")


    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.32, 0.72)

    plt.axis("off")


    plt.title(r"Decision Tree")

    return fig, ax
