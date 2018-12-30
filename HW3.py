# Import Libs
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

p1 = np.array([[772, 780, 589, 477], [404,288, 194, 333]])
p2 = [[360, 515, 736, 438], [566,565, 487, 487]]
p1.shape

A = []
for i in range(0, len(p1[1])):
    x, y = p1[0][i], p1[1][i]
    u, v = p2[0][i], p2[1][i]
    A.append([-1*x, -1*y, -1, 0, 0, 0, u*x, u*y, u])
    A.append([0, 0, 0, -1*x, -1*y, -1, v*x, v*y, v])
A.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
A = np.asarray(A)
U, S, Vh = np.linalg.svd(A)
V = Vh.transpose()
L = Vh[8].reshape((3, 3))

print(pd.DataFrame(A))

sol = [0, 0, 0, 0, 0, 0, 0, 0, 1]
h = np.linalg.solve(A, sol)
print('Is all close? ', np.allclose(np.dot(A, h), sol))

H = np.reshape(h,(3,3))
print("Homography Matrix: ", "\n", H)

player_pos = np.array([589, 194, 1])
print("Position of a player in the transformed Image: ", H@player_pos)

# To validate the result with OpenCV:
p1 = np.array([[772, 780, 589, 477], [404,288, 194, 333]], dtype=np.float32)
p2 = np.array([[360, 515, 736, 438], [566,565, 487, 487]], dtype=np.float32)

p1 = np.array([[772, 404], [780, 288], [589, 194], [477, 333]])
p2 = np.array([[360, 566], [515, 565], [736, 487], [438, 487]])
h, status = cv2.findHomography(p1, p2)
player_pos = np.array([589, 194, 1])
print("Validation of the position of the player with OpenCV's findHomography method: ", h@player_pos)

# Warp first image
im1 = cv2.imread("image1.jpg");
sz = im1.shape
im1 = cv2.warpPerspective(im1, H, (sz[1],sz[0]))

# Player positions
#538, 450
#560, 476

# Left Top Corner is 0,0!

old_points = np.array([[[538, 450]],[[560, 476]]], dtype=np.float32)
new_points = cv2.perspectiveTransform(old_points, H)

w = im1.shape[1]
player1_y = new_points[0, 0, 1]
player2_y = new_points[1, 0, 1]

im1 = cv2.line(im1, (0, player1_y), (w, player1_y), (255, 0, 0), thickness=1, lineType=8)
im1 = cv2.line(im1, (0, player2_y), (w, player2_y), (0, 0, 255), thickness=1, lineType=8)

# Inverse the warp so that the image would return to similar of the first image.
iminvwarped = cv2.warpPerspective(im1, np.linalg.inv(H), (sz[1], sz[0]))
cv2.imwrite('output.png', iminvwarped)

# To Show Warp
plt.imshow(iminvwarped, cmap='gray')
plt.show()


