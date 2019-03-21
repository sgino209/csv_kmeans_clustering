#!/usr/bin/env python
#
# Created by Shahar Gino at March 2019, sgino209@gmail.com
# This program reads data from a csv file, applies kmean, then output the clustering results
# For Usage information, please use the -h flag
#
# Quick Howto:
#
# - Enter Bash shell:
#   % bash
#
# - Enter your virtual environment:
#   % virtualenv -p $(which python) env
#   % source env/bin/activate
#
# - At the first time, install the environment's requirements:
#   % pip install -r requirements.txt
#
# - When done, exit your virtual environment:
# % deactivate

from sklearn.decomposition import PCA
from os                    import path
from time                  import time
from csv                   import reader
from sys                   import argv, stdout
from scipy.cluster.vq      import kmeans, vq, whiten
from getopt                import getopt, GetoptError
from pylab                 import plot, show, grid, get_cmap, legend, title
from numpy                 import vstack, array, asarray, abs, random, linspace

# ------------------------------------------------------------------------------------------------------------

# Python structuring way:
class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# ------------------------------------------------------------------------------------------------------------

def find_nearest(array, value):
    """ Finds the nearest element in a 1D array, for a given value """

    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return array[idx]

# ---------------------------------------------------------------------------------------------------------------

def info(msg, report_file, quiet=False, first_call=False):
    """ Auxiliary method for info printout """

    append_write = 'a'
    if first_call:
        append_write = 'w'

    f = open(report_file, append_write)
    f.write("%s\n" % msg)
    f.close()

    if not quiet:
        print(msg)

# ---------------------------------------------------------------------------------------------------------------

def debug(title, obj, n=10):
    """ Auxiliary method for debug printout """

    top_n = min(n,len(obj))
    print('-'*40)
    print(title)
    print(obj[:top_n])

# ---------------------------------------------------------------------------------------------------------------

def usage():
  """ Usage information """

  script_name = path.basename(__file__)
  print('')
  print('Usage:  %s' % script_name)
  print('')
  print('  Optional Flags |  Default value   |                             Notes                                ')
  print('-----------------|------------------|------------------------------------------------------------------')
  print('  --csv_in       |   input.csv      | Input CSV file (see assumptions below)                           ')
  print('  --report_out   |   output.txt     | Output Report file                                               ')
  print('  --delimiter    |        ;         | Delimiter character for the input CSV                            ')
  print('  --max_clusters |       10         | Maxmimal number of clusters                                      ')
  print('  --blacklist_in |        -         | CSV fields which shall get excluded, delimited by \';\' character')
  print('  --plots_en     |      False       | Enables 2D visualization                                         ')
  print('  --debug_en     |      False       | Enables debug printouts                                          ')
  print('  --quiet        |      False       | Minimal verbosity mode to stdout                                 ')
  print('')
  print('Input CSV format assumptions:')
  print('- First Row is a header')
  print('- First Column holds the subject name, e.g. test_name')
  print('- Fields are delimited by a \';\' character')
  print('')
  print('Usage examples:')
  print('   (-) %s' % script_name)
  print('   (-) %s --csv_in=data_in.csv --reports_out=clusters.txt --blacklist_in=\"time_ms;errors\"' % script_name)
  print('')

# ------------------------------------------------------------------------------------------------------------

def get_user_params(_argv):
  """ Handles user parameter """

  # Defaults:
  args = Struct( csv_in="input.csv",
                 report_out="output.txt",
                 delimiter=";",
                 max_clusters=10,
                 blacklist_in="",
                 plots_en=False,
                 debug_en=False,
                 quiet=False )

  # User-Arguments parameters (overrides defaults):
  try:
      opts, user_args = getopt(_argv, "h", ["csv_in=", "reports_out=", "delimeter=", "max_clusters=", "blacklist_in=", "plots_en", "debug_en", "quiet"])

      for opt, user_arg in opts:
          if opt == '-h':
              usage()
              exit()
          elif opt in "--csv_in":
              args.csv_in = user_arg
          elif opt in "--report_out":
              args.report_out = user_arg
          elif opt in "--delimiter":
              args.delimiter = user_arg
          elif opt in "--max_clusters":
              args.max_clusters = user_arg
          elif opt in "--blacklist_in":
              args.blacklist_in = user_arg
          elif opt in "--plots_en":
              args.plots_en = True
          elif opt in "--debug_en":
              args.debug_en = True
          elif opt in "--quiet":
              args.quiet = True

  except GetoptError, e:
      error(str(e))
      usage()
      exit(2)

  return args

# ------------------------------------------------------------------------------------------------------------

def main(args):
    """ Main function """

    # Init.:
    data_arr  = []
    names_arr = []
    attr_lut  = []
    black_lst = args.blacklist_in.split(";")

    # Data Fetch:
    info('Fetching data...', args.report_out)
    with open(args.csv_in, 'rb') as f:
        f_csv = reader(f, delimiter=args.delimiter)
        header = next(f_csv)[1:]
        for attr in header:
            attr_lut.append({})
        for row in f_csv:
            data_k = []
            names_arr.append(row[0])
            for k in range(len(header)):
                attr = row[k+1].strip()
                if attr != '' and attr in black_lst:
                    info('skipping blacklist field: %s' % attr, args.report_out)
                    continue
                if attr not in attr_lut[k]:
                    attr_lut[k][attr] = len(attr_lut[k])
                data_k.append(float(attr_lut[k][attr]))
            data_arr.append(data_k)

    # Stacking:
    info('Stacking data...', args.report_out)
    data = vstack(data_arr)
    name = vstack(names_arr)

    # Normalization (per feature basis):
    info('Normalizing data...', args.report_out)  # Before running k-means, it is beneficial to rescale each feature dimension of the observation set with whitening.
    features = whiten(data)                       # Each feature is divided by its standard deviation across all observations to give it unit variance.

    # Computing K-Means
    info('Clustering data (K-Means)...', args.report_out)
    max_clusters = min(args.max_clusters,len(name))
    min_err = float('Inf')
    clusters_num = 0
    clusters = []
    for K in range(1,max_clusters):
        centroids, distortion = kmeans(features, K)
        if distortion < min_err:
            min_err = distortion
            clusters_num = K
            clusters = centroids
        if args.debug_en:
            debug('distortion', [K, distortion])

    # Assign a code from a code-book ('clusters') to each observation ('features', must have a unit variance):
    info('Code-Book preparation...', args.report_out)
    code, _ = vq(features, clusters)
    info("using %d clusters, distortion=%.3f" % (clusters_num, min_err), args.report_out)

    # Results (report):
    info('Report Generation...', args.report_out)
    for i in range(clusters_num):
        result_names = name[code==i, 0]
        info("="*50, args.report_out, args.quiet)
        info("Cluster %d" % i, args.report_out, args.quiet)
        cluster = clusters[i]
        cluster_attrs = []
        for dim in range(len(cluster)):
            cluster_dim = cluster[dim]
            features_dim = [x[dim] for x in features]
            data_dim = [x[dim] for x in data]
            nearest_feature = find_nearest(features_dim, cluster_dim)
            nearest_data = data_dim[features_dim.index(nearest_feature)]
            attr = "N/A"
            for k,v in attr_lut[dim].iteritems():
                if v == nearest_data:
                    attr = k
                    break
            cluster_attrs.append("%s~%s" % (header[dim],attr))
            if args.debug_en:
                print("cluster=%d, dim=%d, attr=%s, cluster_dim=%.3f, features_dim=%s, nearst_feature=%.3f, nearest_data=%d, nearest_attr=%s"
                        % (i, dim, header[dim], cluster_dim, str(features_dim), nearest_feature, nearest_data, attr))
        info("Prominent attributes: %s" % str(cluster_attrs), args.report_out, args.quiet)
        info("-"*50, args.report_out, args.quiet)
        for _name in result_names:
            info(_name, args.report_out, args.quiet)

    # Results (2D visualization):
    if args.plots_en:
        info('Visualizing data (with PCA)...', args.report_out)
        pca = PCA(n_components=2)
        pca.fit(features)
        features_pca = pca.transform(features)
        clusters_pca = pca.transform(clusters)
        cmap = get_cmap('gnuplot')
        colors = [cmap(i) for i in linspace(0, 1, clusters_num)]
        for i in range(clusters_num):
            plot(features_pca[code==i,0], features_pca[code==i,1], 'o', color=colors[i], label='Cluster %d' % i)
        plot(clusters_pca[:,0], clusters_pca[:,1], 'sr', markersize=8)
        legend(loc='best')
        title('Clustering Results (PCA domain, 2D)')
        grid()
        show()

    # Debug:
    if args.debug_en:
        print('Name', name)
        print('Data', data)
        print('Clusters', clusters)
        print('Features', features)
        print('Code', code)
        if args.plots_en:
            print('Clusters_pca', clusters_pca)
            print('Features_pca', features_pca)

# ------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    t0 = time()

    args = get_user_params(argv[1:])

    info("Start", args.report_out, False, True)

    main(args)

    t1 = time()
    t_elapsed_sec = t1 - t0
    info("Done! (%.2f sec)" % t_elapsed_sec, args.report_out)

