import tensorflow as tf
import zipfile
import json
import os
import sys
from tensorflow.python.framework import graph_util
# 192.168.1.104 git.deepsight.ai


def load_model(zip_path):
    z = zipfile.ZipFile(zip_path, "r")
    z.extractall()
    name_path = z.namelist()[0].split('/')[0]
    current_path = os.path.join(os.getcwd(), name_path)
    z_list = os.listdir(current_path)
    json_file = [file for file in z_list if file.endswith('json')]
    pb_file = [file for file in z_list if file.endswith('pb')]
    meta_file = [file for file in os.listdir(os.path.join(current_path, 'checkpoints')) if file.endswith('meta')]
    json_data = json.load(open(os.path.join(current_path, json_file[2]), encoding="utf-8"))
    # get alpha and beta
    alpha, beta = json_data['preprocess']['norm']['alpha'], json_data['preprocess']['norm']['beta']

    saver = tf.train.import_meta_graph(os.path.join(current_path, 'checkpoints', meta_file[0]))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, os.path.join('/home/pupa/PycharmProjects/cvalgorithms/export/new_user_dos_sct_0922_sct_20211018183508/checkpoints/SegmentationTF'))
        graph = sess.graph
        with tf.variable_scope("normalize_layer"):
            inputs = tf.get_default_graph().get_tensor_by_name('input_image:0')
            inputs = (inputs - beta) / alpha

            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['pred'])

            with tf.gfile.FastGFile(current_path + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())


    # gd = tf.GraphDef.FromString(open(pb_file[0], "rb").read())
    # inp, predictions = tf.import_graph_def(gd, return_elements=[
    #     "{}:0".format('input_image'), "{}:0".format('pred')])
    # tf.reset_default_graph()
    # # build computer graph
    # images = tf.placeholder(tf.floa32, (None, 224, 224, 3))
    #
    # with tf.variable_scope('normalize_layers'):

    pass


# import tensorflow as tf
#
# model = '/home/pupa/PycharmProjects/cvalgorithms/export/new_user_dos_sct_0922_sct_20211018183508model.pb' #请将这里的pb文件路径改为自己的
# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
# tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('log/', graph)


if __name__ == "__main__":
    zip_path = "/home/pupa/Downloads/20211018183508.zip"
    load_model(zip_path)
