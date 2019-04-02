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
from time                  import time
from csv                   import reader
from pandas                import DataFrame
from sys                   import argv, stdout
from os                    import path, environ
from scipy.cluster.vq      import kmeans, vq, whiten
from getopt                import getopt, GetoptError
from numpy                 import vstack, array, asarray, abs, random, linspace, cumsum, round
from pylab                 import plot, subplot, show, xlabel, ylabel, ylim, grid, get_cmap, legend, title, savefig

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
  print('  Optional Flags  |  Default value   |                             Notes                                                         ')
  print('------------------|------------------|-------------------------------------------------------------------------------------------')
  print('  --csv_in        |   results.csv    | Input CSV file (see assumptions below)                                                    ')
  print('  --delimiter_in  |        ;         | Delimiter character for the input CSV                                                     ')
  print('  --blacklist_in  |        -         | CSV fields which shall get excluded, delimited by \';\' character                         ')
  print('  --report_out    |   buckets.txt    | Output Report file                                                                        ')
  print('  --verbosity_out |        -         | CSV fields which shall be printed out per test in the report, delimited by \';\' character')
  print('  --max_clusters  |       10         | Maxmimal number of clusters                                                               ')
  print('  --plots_en      |      False       | Enables 2D visualization                                                                  ')
  print('  --save_plot_as  |   buckets.png    | 2D visualization filename, if plot is enabled (dpi=100)                                   ')
  print('  --debug_en      |      False       | Enables debug printouts                                                                   ')
  print('  --quiet         |      False       | Minimal verbosity mode to stdout                                                          ')
  print('')
  print('Input CSV format assumptions:')
  print('- First Row is a header')
  print('- First Column holds the subject name, e.g. test_name')
  print('- Fields are delimited by a \';\' character')
  print('')
  print('Usage examples:')
  print('   (-) %s' % script_name)
  print('   (-) %s --csv_in=my_results.csv --reports_out=my_report.txt --blacklist_in=\"time_ms;errors\"' % script_name)
  print('   (-) %s --blacklist_in=\"time_ms;errors;attr1;attr2;attr3;attr4;attr10;attr13;agents_err\" --plots_en --verbosity_out=\"time_ms;errors;agents_err;files_err\" --quiet' % script_name)
  print('')

# ------------------------------------------------------------------------------------------------------------

def get_user_params(_argv):
  """ Handles user parameter """

  # Defaults:
  args = Struct( csv_in="results.csv",
                 delimiter_in=";",
                 blacklist_in="",
                 report_out="buckets.txt",
                 verbosity_out="",
                 max_clusters=10,
                 plots_en=False,
                 save_plot_as="buckets.png",
                 debug_en=False,
                 quiet=False )

  # User-Arguments parameters (overrides defaults):
  try:
      opts, user_args = getopt(_argv, "h", ["csv_in=", "delimeter_in=", "blacklist_in=", "reports_out=", "verbosity_out=", "max_clusters=", "plots_en", "save_plot_as=", "debug_en", "quiet"])

      for opt, user_arg in opts:
          if opt == '-h':
              usage()
              exit()
          elif opt in "--csv_in":
              args.csv_in = user_arg
          elif opt in "--delimiter_in":
              args.delimiter_in = user_arg
          elif opt in "--blacklist_in":
              args.blacklist_in = user_arg
          elif opt in "--report_out":
              args.report_out = user_arg
          elif opt in "--verbosity_out":
              args.verbosity_out = user_arg
          elif opt in "--max_clusters":
              args.max_clusters = int(user_arg)
          elif opt in "--plots_en":
              args.plots_en = True
          elif opt in "--save_plot_as":
              args.save_plot_as = user_arg
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
    header    = []    # header[m] holds the title of the #m data (=#m column in the CSV)
    names_arr = []    # names_arr[k] holds the name of example #k
    code_arr  = []    # code_arr[k][m] holds a the #m code  (=corresponds to header #m value) of example #k 
    data_arr  = []    # data_arr[k][m] holds a the #m value (=corresponds to header #m value) of example #k 
    code_idx  = []    # code_idx[k] maps to the column index of the raw data (taking into account black-list columns) 
    attr_lut  = []    # attr_lut maps from raw values to their corresponding codes (see more details in the comment below)
    
    black_lst = args.blacklist_in.split(";")
    verbosity_lst = args.verbosity_out.split(";")

    # Data Fetch:
    info('Fetching data (skipping: %s)...' % black_lst, args.report_out)
    with open(args.csv_in, 'rb') as f:
        f_csv = reader(f, delimiter=args.delimiter_in)
        
        # Init header and attr_lut:
        header = next(f_csv)
        for k in range(len(header)):
            attr_lut.append({})
            header_attr = header[k].strip()
            if k>0 and header_attr not in black_lst:
                code_idx.append(k) 

        # Main coarse:
        for row in f_csv:
            data_k = []
            code_k = []
            names_arr.append(row[0])
            for k in range(len(header)):
           
                # Store raw data (for a later logging):
                data_k.append(row[k].strip())

                # Populate main data structures (skip blacklist attributes):
                if k in code_idx:
                    attr = row[k].strip()                     # attr_lut[k] holds a dictionary for the k'th column in of the Header, i.e. title of data k'th coloumn.
                    if attr not in attr_lut[k]:               # This dictionary maps from the data original value to its corresponding synthetic code (integer). 
                        attr_lut[k][attr] = len(attr_lut[k])  # For example, in case that the k'th data represents "Car Name" (strings),  
                    code_k.append(float(attr_lut[k][attr]))   # then any "BMW" value will get mapped to '1', any "Mazda" value will get mapped to '2', etc.
            
            data_arr.append(data_k)
            code_arr.append(code_k)

    # Stacking:
    info('Stacking data...', args.report_out)
    data = vstack(code_arr)
    name = vstack(names_arr)

    # Normalization (per feature basis):
    info('Normalizing data...', args.report_out)  # Before running k-means, it is beneficial to rescale each feature dimension of the observation set with whitening.
    features = whiten(data)                       # Each feature is divided by its standard deviation across all observations to give it unit variance.

    # Computing K-Means
    info('Clustering data (K-Means)...', args.report_out)
    max_clusters = min(args.max_clusters+1,len(name))
    min_err = float('Inf')
    clusters_num = 0
    clusters = []
    for K in range(1,max_clusters):
        centroids, distortion = kmeans(features, K)
        if args.debug_en:
            debug('distortion', [K, distortion])
        if distortion < min_err:
            min_err = distortion
            clusters = centroids
            clusters_num = len(clusters)
        else:
            break

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
            op = "~"
            if cluster_dim == nearest_feature:
                op = "="
            for k,v in attr_lut[code_idx[dim]].iteritems():
                if v == nearest_data:
                    attr = k
                    break
            cluster_attrs.append("%s%s%s" % (header[code_idx[dim]],op,attr))
            if args.debug_en:
                print("cluster=%d, dim=%d, attr=%s, cluster_dim=%.3f, nearst_feature=%.3f, nearest_data=%.2f, nearest_attr=%s"
                        % (i, dim, header[code_idx[dim]], cluster_dim, nearest_feature, nearest_data, attr))
        info("Prominent attributes: %s" % str(cluster_attrs), args.report_out, args.quiet)
        info("-"*50, args.report_out, args.quiet)
        for _name in result_names:
            test_name_list = [_name]
            example_idx = names_arr.index(_name)
            for header_attr in verbosity_lst:
                header_idx = header.index(header_attr)
                data_val = data_arr[example_idx][header_idx]
                test_name_list.append("%s=%s" % (header_attr, data_val))
            info(", ".join(test_name_list), args.report_out, True)

    # PCA calculation:
    features_df = DataFrame(features)
    clusters_df = DataFrame(vstack(clusters))
    pca = PCA(n_components=clusters_num)
    pca.fit(features_df)
    components = pca.components_ 
    variance = pca.explained_variance_ratio_       # calculate variance ratios
    var = cumsum(round(variance, decimals=3)*100)  # cumulative sum of variance explained with [n] features
    info("\nPCA Variance: \n%s\n" % str(variance), args.report_out, True) 
    
    # Results (2D visualization):
    if args.plots_en:
        if environ.get('DISPLAY'):
            info('Visualizing data (with PCA)...', args.report_out)
            cmap = get_cmap('gnuplot')
            colors = [cmap(i) for i in linspace(0, 1, clusters_num)]
            
            pca = PCA(n_components=2)
            pca.fit(features_df)
            features_pca = pca.transform(features_df)
            clusters_pca = pca.transform(clusters_df)

            subplot(121)
            for i in range(clusters_num):
                plot(features_pca[code==i,0], features_pca[code==i,1], 'o', color=colors[i], label='Cluster %d' % i)
            plot(clusters_pca[:,0], clusters_pca[:,1], 'sr', markersize=8)
            legend(loc='best')
            xlabel('Principle Component 1')
            ylabel('Principle Component 2')
            title('Clustering Results (PCA domain, 2D)')
            grid()

            subplot(122)
            xlabel('# of Features')
            ylabel('% Variance Explained')
            title('PCA Analysis (clusters contribution)')
            grid()
            plot(var)

            savefig(args.save_plot_as, dpi=100)
            show()
        else:
            info('Skipping Data Visualization since DISPLAY is not defined', args.report_out)

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
