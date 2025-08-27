import numpy as np
import nibabel as nib
from scipy.ndimage import label
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from scipy.stats import wasserstein_distance
from scipy.ndimage import label, generate_binary_structure
from scipy.linalg import sqrtm
from scipy.ndimage import binary_erosion

"""
This class calculates metrics for different data sets and allows to compare the distributions of the
calculated attributes.

Attributes currently calculated are: 
    * ls_count_per_img: number of disconnected features / lesions, per image.
    * all_ls_vxls: The total number of lesion voxels in each image. 
    * lesion_min_size : The minimum size of a lesion in each image.
    * lesion_max_size : The maximum size of a lesion in each image.
    * lesion_mean_size : The mean size of lesions in each image.

    others (not used at the moment because not counted per image, but per lesion or wm is not always available.):):
    * lesion_size. The number of voxels in each seperate feature (calculated not per image, but per lesion.)
    * ls_vxls_in_wm: the number of lesion voxels inside white matter, per image.

""" 

class LesionMetrics:

    def __init__(self,ls_dir, dataset_name, wm_dir=None, ls_suffix=None):
        self.dataset_name = dataset_name
        self.wm_dir = wm_dir
        self.ls_dir = ls_dir
        self.ls_suffix = ls_suffix
        self.img_basenames = []
        
        self.lesion_count_per_img = []
        self.lesion_voxels_per_img = []
        self.lesion_min_size = []
        self.lesion_max_size = []
        self.lesion_mean_size = []
        self.lesion_min_sphericity = []
        self.lesion_max_sphericity = []
        self.lesion_mean_sphericity = []
        self.lesion_min_surface_to_volume_ratio = []
        self.lesion_max_surface_to_volume_ratio = []
        self.lesion_mean_surface_to_volume_ratio = []

        self.lesion_size = []
        self.ls_vxls_in_wm = []
        self.heatmap = None


    def _collect_basenames(self):
        """
        Extracts base names from ls_dir where filenames end with 
        '_source_gan1.nii' or '_target_gan1.nii'. 
        Strips the suffix to get the common basename.
        """
        if self.ls_suffix: 
            suffixes = [self.ls_suffix]
        else:
            suffixes = ["_source_gan1.nii",
                    "_target_gan1.nii", 
                    "_test_gan1_real_A.nii", 
                    "_test_gan1_fake_B_closed_conn3.nii",
                    "_source_gan1.nii.gz", 
                    "_target_gan1.nii.gz",
                    ]

        self.img_basenames = sorted([
            f.name.replace(suffix, "") 
            for f in self.ls_dir.glob("*")
            for suffix in suffixes 
            if f.name.endswith(suffix)
        ])

    def compute_surface_voxels(self, mask):
        """Berechnet die Anzahl der Oberflächen-Voxel in einer Binärmaske."""
        structure = generate_binary_structure(3, 1)
        # Erosion: Voxel, die dabei verschwinden, sind Oberflächen-Voxel
        eroded = np.logical_and(mask, ~binary_erosion(mask, structure=structure))
        return np.count_nonzero(eroded)

    def compute_sphericity(self, mask):
        """
        Berechnet die Sphärizität einer Binärmaske.
        mask: 3D-Bool-Array (Läsion)
        """
        V = np.count_nonzero(mask)
        if V == 0:
            return 0.0
        S = self.compute_surface_voxels(mask)
        if S == 0:
            return 0.0
        sphericity = (np.pi**(1/3)) * (6*V)**(2/3) / S
        return sphericity


    def compute_surface_to_volume_ratio(self, mask):
        """
        Berechnet das Oberfläche-zu-Volumen-Verhältnis einer Binärmaske.
        mask: 3D-Bool-Array (Läsion)
        """
        V = np.count_nonzero(mask)
        if V == 0:
            return 0.0
        S = self.compute_surface_voxels(mask)
        return S / V


    def compute_measures(self):
        """
        loads the wm_masks and lesion_masks for every image and computes the 
        metrics. 
        """

        for name in self.img_basenames: 
            if self.ls_suffix:
                ls_path = self.ls_dir / f"{name}{self.ls_suffix}"
            else:   
                ls_path = self.ls_dir / f"{name}_target_gan1.nii"
            ls_mask = nib.load(ls_path).get_fdata()

            if self.wm_dir is not None:
                wm_mask = nib.load(self.wm_dir / f"{name}_source_gan1.nii").get_fdata()

            # label and count the lesions
            cube = np.ones(shape=(3,3,3))
            labeled_mask, num_lesions = label(input = ls_mask, structure=cube)
            
            # analyse single lesions
            ls_voxels_in_wm = 0
            
            sphericities_lesions = [] 
            svrs_lesions = []
            for i in range(1, int(labeled_mask.max()) + 1):
                lesion = (labeled_mask == i)

                # lesion specific
                ls_size = np.sum(lesion)
                self.lesion_size.append(ls_size)
                
                # accumulated lesion voxels in wm
                if self.wm_dir is not None:
                    ls_voxels_in_wm += np.sum(lesion & (wm_mask > 0))

                # sphericity & sv
                sphericities_lesions.append(self.compute_sphericity(lesion))
                svrs_lesions.append(self.compute_surface_to_volume_ratio(lesion))

            # save results for the image
            self.ls_vxls_in_wm.append(ls_voxels_in_wm)
            self.lesion_voxels_per_img.append(int(np.sum(ls_mask > 0)))
            self.lesion_count_per_img.append(num_lesions)

            self.lesion_min_size.append(int(np.min(self.lesion_size)) if self.lesion_size else 0)
            self.lesion_max_size.append(int(np.max(self.lesion_size)) if self.lesion_size else 0)
            self.lesion_mean_size.append(np.mean(self.lesion_size) if self.lesion_size else 0)

            self.lesion_min_sphericity.append(np.min(sphericities_lesions) if sphericities_lesions else 0)
            self.lesion_max_sphericity.append(np.max(sphericities_lesions) if sphericities_lesions else 0)
            self.lesion_mean_sphericity.append(np.mean(sphericities_lesions) if sphericities_lesions else 0)

            self.lesion_min_surface_to_volume_ratio.append(np.min(svrs_lesions) if svrs_lesions else 0)
            self.lesion_max_surface_to_volume_ratio.append(np.max(svrs_lesions) if svrs_lesions else 0)
            self.lesion_mean_surface_to_volume_ratio.append(np.mean(svrs_lesions) if svrs_lesions else 0)

        print(f"Img {self.img_basenames} measurements computed successfully.")
            

    def plot_boxplot(self, attribute_name):
        """
        Plots a boxplot for the specified attribute.
        """
        if hasattr(self, attribute_name):
            data = getattr(self, attribute_name)
            plt.figure(figsize=(6, 4))
            plt.boxplot(data)
            plt.title(f"Boxplot of {attribute_name}")
            plt.ylabel("Value")
            plt.grid(True)
            plt.show()
        else:
            print(f"Attribut '{attribute_name}' existiert nicht.")


    def plot_histogram(self, attribute_name, bins=20):
        """
        Plots a histogram for the specified attribute.
        """
        if hasattr(self, attribute_name):
            data = getattr(self, attribute_name)
            plt.figure(figsize=(6, 4))
            plt.hist(data, bins=bins, edgecolor='black')
            plt.title(f"Histogram of {attribute_name}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
        else:
            print(f"Attribut '{attribute_name}' existiert nicht.")

    def calc_lesion_heatmap(self, out_path, normalize=True):
        """ 
        sums up all lesion masks and saves a heatmap of the lesions.
        If normalize is True, the heatmap is normalized by the number of images.
        """
        heatmap = None
        ls_img = None

        for name in self.img_basenames:
            if self.ls_suffix:
                ls_path = self.ls_dir / f"{name}{self.ls_suffix}"
            else:   
                ls_path = self.ls_dir / f"{name}_target_gan1.nii"

            img = nib.load(ls_path)
            mask = img.get_fdata()
            if heatmap is None:
                heatmap = np.zeros_like(mask, dtype=np.float32)
                ls_img = img 
            # sum up lesion voxels
            heatmap += mask.astype(np.float32)

        if normalize and len(self.img_basenames) > 0:
            heatmap = heatmap / len(self.img_basenames)

        # save heatmap
        out_path = Path(out_path) / f"{self.dataset_name}_heatmap.nii.gz"
        heatmap_img = nib.Nifti1Image(heatmap, ls_img.affine, ls_img.header)
        nib.save(heatmap_img, out_path)

        self.heatmap = out_path

        print(f"Lesion heatmap saved to {out_path}")

    
    def compare_distributions_emd(self, second_dataset:"LesionMetrics", attributes=None) -> dict: 
        """ 
        Compares the distributions of given attributes between the two datasets. 
        Metric used is the wasserstein distance. Also returns normalized distances 
        (by std of self) and the mean as total_score.
        """ 
        if attributes is None:
            attributes = ['lesion_count_per_img', 'lesion_voxels_per_img', 'lesion_min_size', 'lesion_max_size',
                           'lesion_mean_size', 'lesion_min_sphericity', 'lesion_max_sphericity', 'lesion_mean_sphericity',
                           'lesion_min_surface_to_volume_ratio', 'lesion_max_surface_to_volume_ratio', 'lesion_mean_surface_to_volume_ratio']

        emds = {}
        emds_normed = {}
        for attr in attributes: 
            arr1 = getattr(self, attr)
            arr2 = getattr(second_dataset, attr)
            emd = wasserstein_distance(arr1, arr2)
            emds[attr] = np.round(emd, 4)

            # norm to std of first data set
            std = max(np.std(arr1), 0.0000001)
            emds_normed[attr] = np.round(emd / std)
        emd_score = np.round(np.mean(list(emds_normed.values())), 4)

        return (emds, emds_normed, emd_score)


    def calculate_feature_matrix(self, dataset, attributes):
        """       
        Calculates a feature matrix for the given dataset based on specified attributes.
        Returns a numpy array with shape (n_samples, n_features).
        """
        for attr in attributes:
            if len(getattr(dataset, attr)) != len(dataset.img_basenames):
                raise ValueError(f"Number of attributes measured of '{attr}' do not match the number of images.")

        feature_matrix = []

        for i in range(len(dataset.img_basenames)):
            img_feature_vector = [getattr(dataset, attr)[i] for attr in attributes]
            feature_matrix.append(img_feature_vector)

        return np.array(feature_matrix, dtype=np.float64)
    

    def compare_distributions_fid(self, second_dataset:"LesionMetrics", attributes=None) -> dict:
        """
        Compares the distributions of given attributes between the two datasets.
        Metric used is the Fréchet Inception Distance (FID).
        """
        if attributes is None:
            attributes = [
                            'lesion_count_per_img', 
                            'lesion_voxels_per_img', 
                            'lesion_min_size', 
                            'lesion_max_size', 
                            'lesion_mean_size', 
                            'lesion_min_sphericity', 
                            'lesion_max_sphericity', 
                            'lesion_mean_sphericity',
                            'lesion_min_surface_to_volume_ratio', 
                            'lesion_max_surface_to_volume_ratio', 
                            'lesion_mean_surface_to_volume_ratio']

        # calc FID: 
        X = self.calculate_feature_matrix(self, attributes)
        Y = self.calculate_feature_matrix(second_dataset, attributes)
        mean_X = np.mean(X, axis=0)
        mean_Y = np.mean(Y, axis=0)
        cov_X = np.cov(X, rowvar=False)
        cov_Y = np.cov(Y, rowvar=False)

        # Regularize covariance matrices
        eps = 1e-6
        cov_X += np.eye(cov_X.shape[0]) * eps
        cov_Y += np.eye(cov_Y.shape[0]) * eps

        # sqrtm (matix square root) of the product of covariance matrices
        covmean = sqrtm(cov_X.dot(cov_Y))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = np.sum((mean_X - mean_Y) ** 2) + np.trace(cov_X + cov_Y - 2 * covmean)

        return float(fid)


    def compare_heatmaps(self, second_dataset:"LesionMetrics"):
        """
        Compares the lesion heatmaps of two datasets calculating the mean squared error (MSE).
        """
        if self.heatmap is None or second_dataset.heatmap is None:
            raise ValueError("Heatmaps have not been calculated for one or both datasets.")
        heatmap1 = nib.load(self.heatmap).get_fdata()
        heatmap2 = nib.load(second_dataset.heatmap).get_fdata()

        mse = np.mean((heatmap1 - heatmap2) ** 2)
        return float(mse)

    def results_to_dict(self, emds = None, fid=None, mse_heatmap=None):
        """
        Converts the results of the metrics into a dictionary format.
        """
        if emds: 
            emds, emds_normed, emd_score = emds
        else:
            emds = None
            emds_normed = None
            emd_score = None

        results = {
        "fid": fid,
        "mse_heatmap": mse_heatmap,
        "emd_raw_values": emds,
        "emd_normalized_values": emds_normed,
        "emd_score": emd_score,
        "lesion_count_per_img": self.lesion_count_per_img,
        "lesion_voxels_per_img": self.lesion_voxels_per_img,
        "lesion_min_size": self.lesion_min_size,
        "lesion_max_size": self.lesion_max_size,
        "lesion_mean_size": self.lesion_mean_size,
        "lesion_min_sphericity": self.lesion_min_sphericity,
        "lesion_max_sphericity": self.lesion_max_sphericity,
        "lesion_mean_sphericity": self.lesion_mean_sphericity,
        "lesion_min_surface_to_volume_ratio": self.lesion_min_surface_to_volume_ratio,
        "lesion_max_surface_to_volume_ratio": self.lesion_max_surface_to_volume_ratio,
        "lesion_mean_surface_to_volume_ratio": self.lesion_mean_surface_to_volume_ratio,
    }
        return results
