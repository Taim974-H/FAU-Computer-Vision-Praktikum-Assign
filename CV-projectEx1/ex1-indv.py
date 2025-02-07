import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label, binary_erosion, binary_dilation


def display_image(img, title):
    """Display a 2D image."""
    plt.imshow(img)
    plt.title(title)
    plt.show()

def subsample_cloud(cloud_img, sample_rate=10):
    """Subsample point cloud data for visualization."""
    x_cloud = cloud_img[:, :, 0]
    y_cloud = cloud_img[:, :, 1]
    z_cloud = cloud_img[:, :, 2]
    
    x_cloud_sampled = x_cloud[::sample_rate, ::sample_rate].ravel()
    y_cloud_sampled = y_cloud[::sample_rate, ::sample_rate].ravel()
    z_cloud_sampled = z_cloud[::sample_rate, ::sample_rate].ravel()

    return x_cloud_sampled, y_cloud_sampled, z_cloud_sampled

def plot_2d_cloud(x_cloud, z_cloud):

    plt.figure()
    plt.scatter(x_cloud, z_cloud, c=z_cloud, cmap='viridis', marker='o', alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('2D Point Cloud (X-Z Projection)')
    plt.show()

def plot_3d_cloud(x_cloud, y_cloud, z_cloud):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x_cloud, y_cloud, z_cloud, c=z_cloud, cmap='viridis', marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Z (Depth)')
    plt.title('3D Point Cloud Visualization')
    plt.show()


def fit_plane(p1, p2, p3):
    """
    Computes the equation of a plane given three points in 3D space.

    Parameters:
    p1, p2, p3 (numpy.ndarray): Three points in 3D space, each as an array of shape (3,).

    Returns:
    tuple: (normal, d) where:
        - normal (numpy.ndarray): The normal vector of the plane (A, B, C).
        - d (float): The offset of the plane from the origin (d in nxX + nyY + nzZ = nx = d) 
    
    If the points are collinear (lying on the same line), returns (None, None).

    Steps:
    1. Calculate two vectors that lie in the plane, `v1` and `v2`, by subtracting points.
    2. Calculate the normal vector to the plane using the cross product of `v1` and `v2`.
    3. Check if the points are collinear (normal vector has zero magnitude). If so, return None.
    4. Calculate `d` as the dot product of the normal vector and any one of the points (e.g., `p1`).

    """
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    if np.linalg.norm(normal) == 0:
        return None,None # Skip if points are collinear
    d = np.dot(normal, p1)

    return normal, d

def compute_inliers(points, normal, d, threshold, use_mlesac=False, gamma=2.0):
    """
    Identifies points within a specified distance from a plane.

    Args:
        points (np.ndarray): Points of shape (N, 3)
        normal (np.ndarray): Plane normal vector
        d (float): Plane offset
        threshold (float): Maximum distance for inliers
        use_mlesac (bool): Whether to use MLESAC strategy
        gamma (float): Outlier cost multiplier for MLESAC

    Returns:
        tuple: (inliers mask, cost/negative_inliers)
    """
    distances = np.abs(np.dot(points, normal) - d) / np.linalg.norm(normal)
    
    if use_mlesac:
        # MLESAC: cost based on inlier distances and scaled outlier count
        inliers_mask = distances < threshold
        outlier_distances = distances[~inliers_mask]
        inlier_distances = distances[inliers_mask]
        # Cost is sum of inlier distances + gamma for outliers
        cost = np.sum(inlier_distances) + gamma * len(outlier_distances)

        return inliers_mask, cost
    else:
        # Standard RANSAC: return mask and negative inlier count for maximization
        inliers_mask = distances < threshold
        return inliers_mask, np.sum(inliers_mask)

def run_ransac(cloud_img, threshold, num_iterations, gamma=2.0, use_mlesac=False):
    """
    Runs the RANSAC algorithm to find the best-fitting plane in a 3D point cloud image.

    Parameters:
    - cloud_img (np.ndarray): A 3D array representing the point cloud. If 3D, it has shape (H, W, 3) with (x, y, z) coordinates;
                              if 2D, it has shape (N, 3) with each row representing a 3D point.
    - threshold (float): The distance threshold to consider a point as an inlier to the plane.
    - num_iterations (int): Number of RANSAC iterations to run.

    Returns:
    - best_model (dict): A dictionary containing the 'normal' vector and 'd' value for the best-fit plane.
    - valid_points (np.ndarray): Array of points considered valid (non-zero `z` component) for plane fitting.
    - inliers_list (np.ndarray): Indices of points in valid_points that are inliers to the best model.
    - valid_mask (np.ndarray): A mask indicating which points in cloud_img are valid (non-zero `z` component).
    - inliers_mask (np.ndarray): Boolean mask marking inliers in `valid_points` for the best model.
    """
    max_inliers = -np.inf
    best_model = None
    valid_mask = None
    inliers_list = None
    valid_points = None
    best_cost = np.inf if use_mlesac else -np.inf

    if cloud_img.ndim == 3:
        z_component = cloud_img[:, :, 2]
        valid_mask = z_component != 0
        valid_points = cloud_img[valid_mask].reshape(-1, 3)
    else:
        valid_points = cloud_img

    for _ in range(num_iterations):
        
        indices = np.random.choice(valid_points.shape[0], 3, replace=False) # Randomly pick 3 unique points
        p1, p2, p3 = valid_points[indices]

        normal, d = fit_plane(p1, p2, p3)
        # if normal is None:  # Skip if the plane fitting failed
        #     continue

        # inliers_mask = compute_inliers(valid_points, normal, d, threshold)
        inliers_mask, cost = compute_inliers(valid_points, normal, d, threshold,use_mlesac=use_mlesac, gamma=gamma)
            
        inliers_No = np.sum(inliers_mask)


        if (use_mlesac and cost < best_cost) or (not use_mlesac and inliers_No > max_inliers):
            # If current model has more inliers, update the best model
            best_model = {
                'normal': normal,
                'd': d
            }
            best_cost = cost
            max_inliers = inliers_No
            inliers_list = np.where(inliers_mask)[0]

    return best_model, valid_points, inliers_list, valid_mask, inliers_mask

def create_initial_mask(img_shape, inliers_list, valid_mask):
    
    mask_img = np.zeros(img_shape[:2], dtype=np.uint8) # Initialize with the original image shape
    valid_indices = np.where(valid_mask.flatten())[0] # Flatten valid_mask to map inlier indices back to the 2D image space
    floor_indices = valid_indices[inliers_list]
    mask_img.flat[floor_indices] = 1  # Set floor pixels to 1

    return mask_img

def refine_mask(mask_img):
    # morphological operations
    # instead of using cv2.erode and cv2.dilate, we can use cv2.morphologyEx with MORPH_CLOSE and MORPH_OPEN
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)  # Close small holes (erosion followed by dilation)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)  # Remove small objects (dilation followed by erosion)

    return refined_mask

def visualize_mask(cloud_img,initial_mask, refined_mask):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display initial mask
    axs[0].imshow(initial_mask, cmap='gray')
    axs[0].set_title("Initial Mask")

    # Display refined mask
    axs[1].imshow(refined_mask, cmap='gray')
    axs[1].set_title("Refined Mask")

    # # 3D visualization of the points classified as floor
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # floor_points = cloud_img[initial_mask == 1]
    # ax.scatter(floor_points[:, 0], floor_points[:, 1], floor_points[:, 2], c='blue', s=1)
    # plt.title("3D Visualization of Floor Points")
    # plt.show()
    plt.show()



def find_largest_connected_component(binary_mask):
    """
    Identifies the largest connected component in a binary mask.

    Parameters:
    binary_mask (numpy.ndarray): A binary image (2D array) where the foreground is represented by 1s (True) and the background by 0s (False).
                               This image can contain multiple connected components, and the function will return the largest one.

    Returns:
    numpy.ndarray: A binary mask where only the largest connected component is retained (1s), and all other areas (including the background) are set to 0s (False).

    Steps:
    1. Label the connected components in the binary mask.
    2. Count the size of each connected component.
    3. Identify the largest component (excluding the background).
    4. Generate a mask that contains only the largest connected component.
    """
    labeled_mask, num_features = label(binary_mask)
    component_sizes = np.bincount(labeled_mask.ravel())   # Count the size of each component
    largest_component_index = np.argmax(component_sizes[1:]) + 1  #(ignore index 0, which is the background) +1 to offset for background index
    largest_component_mask = (labeled_mask == largest_component_index)

    return largest_component_mask


def are_planes_parallel(normal1, normal2, threshold_angle=5.0):
    # Normalize the normal vectors
    n1 = normal1 / np.linalg.norm(normal1)
    n2 = normal2 / np.linalg.norm(normal2)
    # Calculate the angle using the dot product
    cos_theta = np.dot(n1, n2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical inaccuracies
    angle_deg = np.degrees(angle_rad)
    # Check if the angle is within the threshold
    is_parallel = angle_deg < threshold_angle or angle_deg > (180 - threshold_angle)

    return is_parallel, angle_deg

def calculate_distance_between_planes(normal1, d1, normal2, d2):
    
    numerator = abs(d2 - d1)
    normal_magnitude = np.linalg.norm(normal1)
    distance = numerator / normal_magnitude

    return distance

def mask_dimensions(box_mask, cloud_img):
    # Extract 3D points that belong to the box using the mask
    box_points = cloud_img[box_mask]

    # If there are no points in the box, return None
    if box_points.shape[0] == 0:
        return None

    # Find the minimum and maximum points
    min_x = np.min(box_points[:, 0])
    max_x = np.max(box_points[:, 0])
    min_y = np.min(box_points[:, 1])
    max_y = np.max(box_points[:, 1])

    length = max_x - min_x  # Length along the x-axis
    width = max_y - min_y    # Width along the y-axis

    return length, width

def visualize_box_corners(corners):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='blue', s=100, label='Box Corners', marker='o')
    # Draw lines between corners to form the box
    edges = [
        [corners[0], corners[1]], [corners[1], corners[2]], [corners[2], corners[3]], [corners[3], corners[0]],  # Bottom face
        [corners[4], corners[5]], [corners[5], corners[6]], [corners[6], corners[7]], [corners[7], corners[4]],  # Top face
        [corners[0], corners[4]], [corners[1], corners[5]], [corners[2], corners[6]], [corners[3], corners[7]]   # Vertical edges
    ]
    for edge in edges:
        ax.plot(*zip(*edge), color='blue', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Box Corners')
    ax.legend()
    ax.view_init(elev=20, azim=30)  # Set view angle
    plt.show()

    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////

def preemptive_ransac(cloud_img, num_hypotheses=100, batch_size=100, threshold=0.05, use_mlesac=False, gamma=2.0):
    """
    Implements Preemptive RANSAC algorithm.
    
    Args:
    - cloud_img: Input point cloud
    - num_hypotheses: Total number of initial hypotheses (M)
    - batch_size: Number of points to evaluate before preemption (B)
    - threshold: Inlier distance threshold
    - use_mlesac: Whether to use MLESAC cost function
    - gamma: Outlier cost multiplier for MLESAC
    
    Returns:
    - Best plane model
    - Inliers list
    - Valid points
    - Valid mask
    - Inliers mask
    """
    def preemption_function(i, M, B):
        """
        Calculates number of top hypotheses to retain at iteration i
        
        Args:
        - i: Current iteration
        - M: Total initial hypotheses
        - B: Batch size
        
        Returns:
        Number of top hypotheses to retain
        """
        return max(1, int(M * (1 - i / (cloud_img.shape[0] / B))))

    # Prepare valid points
    if cloud_img.ndim == 3:
        z_component = cloud_img[:, :, 2]
        valid_mask = z_component != 0
        valid_points = cloud_img[valid_mask].reshape(-1, 3)
    else:
        valid_points = cloud_img
        valid_mask = np.ones(cloud_img.shape[0], dtype=bool)

    # Generate initial hypotheses
    hypotheses = []
    for _ in range(num_hypotheses):
        indices = np.random.choice(valid_points.shape[0], 3, replace=False)
        p1, p2, p3 = valid_points[indices]
        
        normal, d = fit_plane(p1, p2, p3)
        if normal is None:
            continue
        
        hypotheses.append((normal, d))

    # Preemptive evaluation
    for i in range(0, valid_points.shape[0], batch_size):
        batch = valid_points[i:i+batch_size]
        
        # Evaluate hypotheses on current batch
        hypothesis_scores = []
        for normal, d in hypotheses:
            if use_mlesac:
                _, cost = compute_inliers(batch, normal, d, threshold, use_mlesac=True, gamma=gamma)
                hypothesis_scores.append(cost)
            else:
                inliers_mask, inliers_count = compute_inliers(batch, normal, d, threshold)
                hypothesis_scores.append(-inliers_count)  # Negative for maximization

        # Sort hypotheses by score
        sorted_indices = np.argsort(hypothesis_scores)
        
        # Retain top hypotheses based on preemption function
        top_k = preemption_function(i, num_hypotheses, batch_size)
        hypotheses = [hypotheses[idx] for idx in sorted_indices[:top_k]]

    # Final best hypothesis
    best_normal, best_d = hypotheses[0]
    
    # Compute final inliers
    inliers_mask, _ = compute_inliers(valid_points, best_normal, best_d, threshold, use_mlesac=use_mlesac, gamma=gamma)
    
    best_model = {
        'normal': best_normal,
        'd': best_d
    }
    
    return best_model, valid_points[inliers_mask], np.where(inliers_mask)[0], valid_mask, inliers_mask

def main():

    x = 2
    file_path = f'CV-projectEx1/data/example{x}kinect.mat'
    data = scipy.io.loadmat(file_path)
    amplitude_img, distance_img, cloud_img = data[f'amplitudes{x}'], data[f'distances{x}'], data[f'cloud{x}']


    # Experiment with different M and B values
    configurations = [
        (50, 50),   # Conservative: fewer hypotheses, smaller batches
        (100, 100), # Balanced approach
        (200, 200)  # Aggressive: more hypotheses, larger batches
    ]

    plt.figure(figsize=(15, 5))
    for i, (M, B) in enumerate(configurations, 1):
        # Run Preemptive RANSAC
        floor_model, valid_points, floor_inliers, valid_mask, inliers_mask = preemptive_ransac(
            cloud_img, 
            num_hypotheses=M, 
            batch_size=B, 
            threshold=0.05
        )
        
        # Create and visualize mask
        initial_mask = create_initial_mask(cloud_img.shape, floor_inliers, valid_mask)
        refined_mask = refine_mask(initial_mask)
        
        plt.subplot(1, 3, i)
        plt.imshow(refined_mask, cmap='gray')
        plt.title(f'M={M}, B={B}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Optional: Compute metrics for comparison
    for M, B in configurations:
        floor_model, valid_points, floor_inliers, valid_mask, inliers_mask = preemptive_ransac(
            cloud_img, 
            num_hypotheses=M, 
            batch_size=B, 
            threshold=0.05
        )
        print(f"M={M}, B={B}: {len(floor_inliers)} inliers")


    # Display amplitude and distance images
    display_image(amplitude_img, 'Amplitude Image')
    display_image(distance_img, 'Distance Image')

    # Subsample the point cloud
    x_cloud_sampled, y_cloud_sampled, z_cloud_sampled = subsample_cloud(cloud_img, sample_rate=10)
    # Plot 2D and 3D point clouds
    plot_2d_cloud(x_cloud_sampled, z_cloud_sampled)
    plot_3d_cloud(x_cloud_sampled, y_cloud_sampled, z_cloud_sampled)

    # RANSAC to find the first plane (floor)
    floor_model, valid_points, floor_inliers,valid_mask,inliers_mask = run_ransac(cloud_img, num_iterations=3000, threshold=0.05)
    

    # MLESAC to find the first plane (floor) 
    floor_model_mlesac, valid_points, floor_inliers_mlesac, valid_mask, inliers_mask_mlesac = run_ransac(cloud_img, num_iterations=3000, threshold=0.05, use_mlesac=True, gamma=2)
    
    # Compare inliers
    print(f"RANSAC floor inliers: {len(floor_inliers)}")
    print(f"MLESAC floor inliers: {len(floor_inliers_mlesac)}")
   
    # print(f"Best floor model found with {len(floor_inliers)} inliers.")
    floor_model['normal'] = floor_model['normal'] / np.linalg.norm(floor_model['normal'])
    floor_model_mlesac['normal'] /= np.linalg.norm(floor_model_mlesac['normal'])
   
    initial_mask = create_initial_mask(cloud_img.shape, floor_inliers, valid_mask) # floor pixels are set to 1
    refined_mask = refine_mask(initial_mask)

    #-----------------------------------------------------------------------------------------------

    # Extract non-floor points using the floor mask
    
    floor_inliers_mask = np.zeros(cloud_img.shape[:2], dtype=bool)
    flat_indices = np.where(valid_mask.ravel())[0][floor_inliers]  # Find indices of floor points
    floor_inliers_mask.ravel()[flat_indices] = True  # Update mask to reflect floor inliers (mark the positions in the mask that correspond to the floor points as inliers (True))

    non_floor_cloud_img = cloud_img.copy()
    non_floor_cloud_img[floor_inliers_mask] = 0 
    print(f"Non-floor cloud shape: {non_floor_cloud_img.shape}")

    # RANSAC to find the second plane (top)
    top_model, valid_points, top_inliers, valid_top_mask, inliers_mask = run_ransac(non_floor_cloud_img, num_iterations=3000, threshold=0.05)
    print(f"Best top model found with {len(top_inliers)} inliers.")
    top_model['normal'] = top_model['normal'] / np.linalg.norm(top_model['normal'])

    # Find top plane with MLESAC
    top_model_mlesac, _, top_inliers_mlesac, _, _ = run_ransac(non_floor_cloud_img, num_iterations=3000, threshold=0.05,use_mlesac=True, gamma=2 )
    top_model_mlesac['normal'] /= np.linalg.norm(top_model_mlesac['normal'])

    # Compare inliers
    print(f"RANSAC top plane inliers: {len(top_inliers)}")
    print(f"MLESAC top plane inliers: {len(top_inliers_mlesac)}")

    # RANSAC top plane processing
    initial_top_mask = create_initial_mask(cloud_img.shape, top_inliers, valid_top_mask)
    refined_top_mask = refine_mask(initial_top_mask)
    largest_top_component_mask = find_largest_connected_component(refined_top_mask)

    # MLESAC top plane processing
    initial_top_mask_mlesac = create_initial_mask(cloud_img.shape, top_inliers_mlesac, valid_top_mask)
    refined_top_mask_mlesac = refine_mask(initial_top_mask_mlesac)
    largest_top_component_mask_mlesac = find_largest_connected_component(refined_top_mask_mlesac)


    #-----------------------------------------------------------------------------------------------

    # Visualization comparison
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # RANSAC Visualization
    axs[0, 0].imshow(initial_top_mask, cmap='gray')
    axs[0, 0].set_title("RANSAC Initial Top Mask")

    axs[0, 1].imshow(refined_top_mask, cmap='gray')
    axs[0, 1].set_title("RANSAC Refined Top Mask")

    # MLESAC Visualization
    axs[1, 0].imshow(initial_top_mask_mlesac, cmap='gray')
    axs[1, 0].set_title("MLESAC Initial Top Mask")

    axs[1, 1].imshow(refined_top_mask_mlesac, cmap='gray')
    axs[1, 1].set_title("MLESAC Refined Top Mask")

    # Largest Connected Components
    axs[2, 0].imshow(largest_top_component_mask, cmap='gray')
    axs[2, 0].set_title("RANSAC Largest Connected Top Mask")

    axs[2, 1].imshow(largest_top_component_mask_mlesac, cmap='gray')
    axs[2, 1].set_title("MLESAC Largest Connected Top Mask")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Comparative overlay
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(refined_mask, cmap='Blues', alpha=0.6)
    plt.imshow(largest_top_component_mask, cmap='Reds', alpha=0.5)
    plt.title("RANSAC: Floor and Top Masks")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(refined_mask, cmap='Blues', alpha=0.6)
    plt.imshow(largest_top_component_mask_mlesac, cmap='Reds', alpha=0.5)
    plt.title("MLESAC: Floor and Top Masks")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate distances and dimensions for both
    distance_ransac = calculate_distance_between_planes(
        floor_model['normal'], floor_model['d'], 
        top_model['normal'], top_model['d']
    )
    distance_mlesac = calculate_distance_between_planes(
        floor_model['normal'], floor_model['d'], 
        top_model_mlesac['normal'], top_model_mlesac['d']
    )

    print(f"RANSAC Plane Distance: {distance_ransac:.6f}")
    print(f"MLESAC Plane Distance: {distance_mlesac:.6f}")

    # Dimensions comparison
    dimensions_ransac = mask_dimensions(largest_top_component_mask, cloud_img)
    dimensions_mlesac = mask_dimensions(largest_top_component_mask_mlesac, cloud_img)

    if dimensions_ransac and dimensions_mlesac:
        length_ransac, width_ransac = dimensions_ransac
        length_mlesac, width_mlesac = dimensions_mlesac
        
        print(f"RANSAC - Length: {length_ransac:.2f}, Width: {width_ransac:.2f}")
        print(f"MLESAC - Length: {length_mlesac:.2f}, Width: {width_mlesac:.2f}")


if __name__ == "__main__":
    main()
