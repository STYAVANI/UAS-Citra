import numpy as np

def k_means_segmentation(image_data, k, max_iterations=100):
    # Inisialisasi pusat cluster secara acak
    # Randomly initialize cluster centers
    num_samples, num_features = image_data.shape
    cluster_centers = image_data[np.random.choice(num_samples, k, replace=False)]
    
    for _ in range(max_iterations):
        # Hitung jarak titik data dari pusat cluster
        # Compute distances from data points to cluster centers
        distances = np.sqrt(((image_data - cluster_centers[:, np.newaxis])**2).sum(axis=2))
        
        # Tetapkan kembali titik data ke klaster terdekat
        # Assign data points to nearest cluster
        labels = np.argmin(distances, axis=0)
        
        # Hitung pusat cluster yang baru
        # Compute new cluster centers
        new_centers = np.array([image_data[labels == i].mean(axis=0) for i in range(k)])
        
        # Cek konvergensi
        # Check for convergence
        if np.allclose(cluster_centers, new_centers):
            break
        
        cluster_centers = new_centers
    
    return labels, cluster_centers

# Contoh penggunaan
# Example usage
image_data = np.random.rand(100, 3)  # Data citra dengan 100 titik dan 3 fitur
k = 4  # Jumlah cluster yang diinginkan
labels, centers = k_means_segmentation(image_data, k)

print(labels)
print(centers)