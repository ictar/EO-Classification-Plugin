import numpy as np
import matplotlib.pyplot as plt

# reference: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html

def tmp():
    # define three cluster centers
    centers = [
        [4,2],
        [1,7],
        [5,6],
    ]

    # define three cluster sigmas in x and y, respectively
    sigmas = [
        [0.8, 0.3],
        [0.3, 0.5],
        [1.1, 0.7],
    ]

    # generate test data
    np.random.seed(42) # set seed for reproducibility

    xpts = np.zeros(1)
    ypts = np.zeros(1)
    labels = np.zeros(1)

    data_num_per_cluster = 200
    for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
        xpts = np.hstack((xpts, np.random.standard_normal(data_num_per_cluster) * xsigma + xmu))
        ypts = np.hstack((ypts, np.random.standard_normal(data_num_per_cluster) * ysigma + ymu))
        labels = np.hstack((labels, np.ones(data_num_per_cluster)*i))

    print(xpts.shape, ypts.shape, labels.shape)
    dim, cluster_num, data_num = 2, len(sigmas), xpts.shape[0]
    xpts = xpts.reshape((data_num,1))
    ypts = ypts.reshape((data_num,1))
    labels = labels.reshape((data_num,1))
    # save
    dataset = np.concatenate((xpts, ypts, labels), 1)
    fname = "data_{}_{}_{}.txt".format(dim, cluster_num, data_num)
    np.savetxt(fname, dataset)

    # visualize the test data
    fig0, ax0 = plt.subplots()
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
    for label in range(cluster_num):
        ax0.plot(xpts[labels==label], ypts[labels == label], '.', color=colors[label])
    ax0.set_title("Test data: 200 points x{} clusters.".format(cluster_num))
    plt.show()

""""
Data simulation base on a stochastic model

INPUT:
  Ntot  = total number of observations to be generated        (1 x 1)
  Nts   = number of training sample data for each cluster     (1 x 1)
  Ncl   = number of clusters                                  (1 x 1)
  Nb    = number of bands                                     (1 x 1)
  mu_cl = model means of clusters              (optional)     (Ncl x Nb)
  C_cl  = model cov. Matrices of cluster       (optional)     (Ncl x Nb x Nb)
   p_cl  = model probabaility of clusters       (optional)     (Ncl x 1)

OUTPUT:
  mix   = simulated observation                               (Ntot x Nb)
  label = labels                                              (Ntot x 1)
  ts    = training samples                                    (Ncl x Nts x Nb)
  mu_cl = model means of clusters                             (Ncl x Nb)
  C_cl  = model cov. Matrices of clusters                     (Ncl x Nb x Nb)
  p_cl  = model probabaility of clusters                      (Ncl x 1)
  mu_ts = estimated mean of clusters                          (Ncl x Nb)
  C_ts  = estimated cov. Matrices of clusters                 (Ncl x Nb x Nb)
  p_ts  = estimated probability of clusters                   (Ncl x 1)
"""
def simudata(Ntot, Nts, Ncl, Nb, mu_cl, C_cl, p_cl):
    pass

try:
    from osgeo import gdal
    from osgeo import gdalnumeric
    from osgeo import gdal_array
    from osgeo import osr
except ImportError:
    import gdal

NODATA = -9999   
def write_array_to_raster_multiband(data, save_to, geotransform, SRID=4326):
    
        driver = gdal.GetDriverByName('GTiff')

        nbands, rows, cols = data.shape
        # create a new raster data source
        dataset = driver.Create(
            save_to,
            cols, rows,
            nbands, gdal.GDT_Float32,
        )

        dataset.SetGeoTransform(geotransform)

        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(SRID)

        dataset.SetProjection(out_srs.ExportToWkt())

        for i in range(nbands):
            dataset.GetRasterBand(i+1).WriteArray(data[i])
            dataset.GetRasterBand(i+1).SetNoDataValue(NODATA)

        # close raster file
        dataset = None

if __name__ == '__main__':
    from scipy.io import loadmat
    dname = r'performance/data_10000.mat'
    mat = loadmat(dname)
    dataset = mat['mix']
    write_array_to_raster_multiband(dataset, "data_10000.tiff")