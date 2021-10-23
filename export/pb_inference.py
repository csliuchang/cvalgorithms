import tensorflow as tf
import numpy as np

def pb_infer(pb_model_file, image_np, input_tensor_name='input_image', output_tensor_name='pred'):
    gd = tf.GraphDef.FromString(open(pb_model_file, "rb").read())
    inp, predictions = tf.import_graph_def(gd, return_elements=[
        "{}:0".format(input_tensor_name), "{}:0".format(output_tensor_name)])

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(graph=inp.graph, config=config) as sess:
        pred = predictions.eval(feed_dict={inp: image_np})
        sess.close()
    tf.reset_default_graph()

    return pred


if __name__ == "__main__":
    pb_file = '/home/pupa/PycharmProjects/cvalgorithms/export/new_user_dos_sct_0922_sct_20211018183508/new_user_dos_sct_0922_sct_20211018183508_fix_size.pb.cvt'
    image_np = np.ones((1, 448, 256, 3), dtype=np.float32) * 128
    pred = pb_infer(pb_file, image_np)
    pb_file_g = '/home/pupa/PycharmProjects/cvalgorithms/export/new_user_dos_sct_0922_sct_20211018183508/new_user_dos_sct_0922_sct_20211018183508_fix_size.pb'
    pred_g = pb_infer(pb_file_g, image_np)
    min, max, mean = np.min(pred_g-pred), np.max(pred_g-pred), np.mean(pred_g-pred)
    print(min, max, mean)