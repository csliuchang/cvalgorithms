from tensorflow.python.platform import gfile
import tensorflow as tf

model = 'C:/Users/user1/Desktop/classify/20210106193629_fix_size.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/', graph)