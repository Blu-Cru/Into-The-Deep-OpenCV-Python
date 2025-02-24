import cv2
import numpy as np
import math

def main():
    src = cv2.imread(r"images\chart\1.jpg")
    if src is None:
        print("Error: image not loaded")
        
    show("Src", src)

    undistorted = undistort(src)
    show("Undistorted", undistorted)

    K = np.matrix([
        [1279.33,       0, 958.363], 
        [      0, 1279.33, 492.062], 
        [      0,       0,       1]
    ])

    img_points = [
        (895, 607), (1155, 602),
        (870, 810), (1207, 805)
    ]

    points = undistorted.copy()

    for (x, y) in img_points:
            print(x, y)
            cv2.circle(points, (int(x), int(y)), 5, (0, 0, 255), -1)
            
    show("Points", points)

    np_img_points = np.float32([[895, 607], [1155, 602],
        [870, 810], [1207, 805]])
    np_top_down_points = np.float32([[800, 600], [1000, 600],
        [800, 800], [1000, 800]])
    
    M = cv2.getPerspectiveTransform(np_img_points,
                                    np_top_down_points)
    
    # each inch is 40 pixels
    top_down_size = (1920, 1080)  # (width, height)
    top_down_view = cv2.warpPerspective(undistorted, M, top_down_size)

    # -------------------------------------------------------
    # 8) Save or display the result
    # -------------------------------------------------------
    show("Top-Down View", top_down_view)

    # # 2) Get the rotation for pitch=-25 deg, yaw=0, roll=0
    # pitch_deg = -65.0
    # R1 = compute_rotation_matrix_pitch_yaw_roll(pitch_deg, 0.0, 0.0)
    # R2 = compute_rotation_matrix_pitch_yaw_roll(0.0, pitch_deg, 0.0)
    # R3 = compute_rotation_matrix_pitch_yaw_roll(0.0, 3.0, pitch_deg)

    # # 3) Assume the camera is located at (0, 0, 322) in the WORLD frame
    # #    i.e., 322 mm above the plane z=0.
    # camera_center = np.array([0.0, 0.0, 12.6 - 1.5])

    # # 4) Compute the homography H
    # H1 = compute_homography(K, R1, camera_center)
    # H2 = compute_homography(K, R2, camera_center)
    # H3 = compute_homography(K, R3, camera_center)

    # print("Intrinsic matrix K:\n", K)
    # print("\nRotation matrix R (pitch=-25°, yaw=0°, roll=0°):\n", R3)
    # print("\nCamera center:\n", camera_center)
    # print("\nHomography H:\n", H3)

    # # 5) Apply H to some example points on the ground plane
    # points_on_plane = np.array([
    #     [  4.0,   13.0],   # the origin
    #     [4.0,   18.0],
    #     [-4.0, 18.0],
    #     [  -4.0, 13.0],
    #     [-10.0, 18.0]
    # ])
    # print("points on plane:\n", points_on_plane)

    # # image_points1 = apply_homography(H1, points_on_plane)
    # # image_points2 = apply_homography(H2, points_on_plane)
    # image_points3 = apply_homography(H3, points_on_plane)

    # # # Print out the resulting pixel coordinates
    # # print("H1\nProjected image points:\n", image_points1)
    # # print("H2\nProjected image points:\n", image_points2)
    print("H3\nProjected image points:\n", image_points3)
    list_of_tuples = list(map(tuple, image_points3))

    clean_list = []

    for item in list_of_tuples:
        # item is (matrix([[x, y]]),)
        mat = item[0]      # 'mat' is a numpy.matrix of shape (1, 2)
        arr = np.asarray(mat)   # convert to a standard array, shape (1,2)
        # now extract x, y
        x, y = arr[0, 0], arr[0, 1]
        clean_list.append((x, y))

    # Now 'clean_list' is a list of (x, y) tuples
    for (x, y) in clean_list:
        print(x, y)
        cv2.circle(undistorted, (int(x), int(y)), 5, (0, 0, 255), -1)

    show("Points", undistorted)

    # # -------------------------------------------------------
    # # 5) Define a rectangle on the plane that we want to view
    # #    in top-down coordinates. Example: 400mm x 400mm area.
    # #    We’ll assume (0,0) is directly under the camera, but
    # #    you can pick any region you like.
    # # -------------------------------------------------------
    # plane_corners = np.array([
    #     [20.0,   35.0],
    #     [-20.0, 35.0],
    #     [-20.0, 10.0],
    #     [20.0,   10.0]
    # ], dtype=np.float32)

    # # Project these plane corners into the image via H
    # image_corners = apply_homography(H3, plane_corners)

    # # -------------------------------------------------------
    # # 6) Specify the “destination” corners in the top-down view.
    # #    For a 1 pixel/mm mapping, we can make the output
    # #    image 400 x 400 pixels.
    # # -------------------------------------------------------
    # top_down_corners = np.array([
    #     [0.0,   0.0],
    #     [600.0, 0.0],
    #     [600.0, 600.0],
    #     [0.0,   600.0]
    # ], dtype=np.float32)

    # # -------------------------------------------------------
    # # 7) Use OpenCV to get a perspective transform and warp
    # # -------------------------------------------------------
    # M = cv2.getPerspectiveTransform(image_corners.astype(np.float32),
    #                                 top_down_corners.astype(np.float32))
    # print("M", M)

    # top_down_size = (600, 600)  # (width, height)
    # top_down_view = cv2.warpPerspective(undistorted, M, top_down_size)

    # # -------------------------------------------------------
    # # 8) Save or display the result
    # # -------------------------------------------------------
    # cv2.imwrite("top_down_result.jpg", top_down_view)
    # cv2.imshow("Top-Down View", top_down_view)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

def compute_rotation_matrix_pitch_yaw_roll(pitch_deg, yaw_deg=0.0, roll_deg=0.0):
    """
    Returns the 3x3 rotation matrix given pitch, yaw, roll in degrees.
    Convention: Rz(yaw) * Ry(pitch) * Rx(roll)
    (You can adapt if your convention differs, but below uses Y-axis for pitch.)
    """
    # Convert degrees to radians
    pitch = np.deg2rad(pitch_deg)
    yaw   = np.deg2rad(yaw_deg)
    roll  = np.deg2rad(roll_deg)
    
    # For clarity, define sines/cosines
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw),   np.cos(yaw)
    sr, cr = np.sin(roll),  np.cos(roll)
    
    # Rotation about y-axis (pitch)
    Ry = np.array([
        [ cp,  0.,  sp ],
        [ 0.,  1.,  0. ],
        [-sp,  0.,  cp ]
    ])
    
    # Rotation about z-axis (yaw)
    Rz = np.array([
        [ cy, -sy,  0. ],
        [ sy,  cy,  0. ],
        [ 0.,  0.,  1. ]
    ])
    
    # Rotation about x-axis (roll)
    Rx = np.array([
        [ 1.,   0.,   0. ],
        [ 0.,   cr,  -sr ],
        [ 0.,   sr,   cr ]
    ])
    
    # Combined rotation (order: yaw -> pitch -> roll)
    # Adjust if your coordinate system demands a different order
    R = Rz @ Ry @ Rx
    
    return R

def compute_homography(K, R, camera_center):
    """
    Computes the 3x3 homography H that maps a point (X, Y, 1) on the z=0 plane
    in the world frame to the image plane, given K, R, and camera_center in world coords.
    """
    # camera_center is a 3D vector [Cx, Cy, Cz] in the world frame
    # translation (in the camera’s extrinsic) is t = -R @ C
    t = -R @ camera_center
    
    # Extract r1 and r2 (the first two columns of R),
    # then form a 3x3 by [r1, r2, t]
    r1 = R[:, 0]
    r2 = R[:, 1]
    # Stack them side by side into a 3x3
    R_2cols_t = np.column_stack((r1, r2, t))
    
    # Finally multiply by K to get the homography
    H = K @ R_2cols_t
    return H

def apply_homography(H, points_2D):
    """
    Applies the 3x3 homography H to a set of 2D points (on z=0 plane).
    
    points_2D should be an array of shape (N, 2), each row = [X, Y].
    Returns projected pixel coordinates of shape (N, 2).
    """
    # Convert to homogeneous
    num_pts = points_2D.shape[0]
    hom_pts = np.column_stack([points_2D, np.ones(num_pts)])
    
    # Transform
    projected = (H @ hom_pts.T).T  # shape (N, 3)
    
    # Normalize to get pixel coords
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    
    return projected[:, :2]
    
def undistort(img):
    h, w = img.shape[:2]

    cameraMatrix = np.array([[1279.33,   0, 958.363], 
                             [  0, 1279.33, 492.062], 
                             [  0,   0,   1]], dtype=np.float64)
    
    distCoeffs = np.array([-0.448017, 0.245668, -0.000901464, 0.000996399], dtype=np.float64)

    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, 
        distCoeffs, 
        (w, h), 
        alpha=1, 
        newImgSize=(w, h)
    )

    return cv2.undistort(img, cameraMatrix, distCoeffs, None, newCameraMatrix)
    
def show(str, mat):
    cv2.imshow(str, cv2.resize(mat, (960, 540)))

main()