import argparse
import sys
import onnx


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='Path of directory saved the input model.')
    parser.add_argument('--original', required=True, nargs='+', help='The original name you want to modify.')
    parser.add_argument('--new_name', required=True, nargs='+', help='The new name you want change to, the number of new name should be same with the number of origin_names')
    parser.add_argument('--save_dir', required=True, help='Path to save the new onnx model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model = onnx.load(args.onnx)
    output_tensor_names = set()
    for ipt in model.graph.input:
        output_tensor_names.add(ipt.name)
    for node in model.graph.node:
        for out in node.output:
            output_tensor_names.add(out)

    for origin_name in args.original:
        if origin_name not in output_tensor_names:
            print("[ERROR] Cannot find tensor name '{}' in onnx model graph.".format(origin_name))
            sys.exit(-1)
    if len(set(args.original)) < len(args.original):
        print("[ERROR] There's dumplicate name in --on, which is not allowed.")
        sys.exit(-1)
    if len(args.new_name) != len(args.original):
        print("[ERROR] Number of --nn must be same with the number of --on.")
        sys.exit(-1)
    if len(set(args.new_name)) < len(args.new_name):
        print("[ERROR] There's dumplicate name in --nn, which is not allowed.")
        sys.exit(-1)
    for new_name in args.new_name:
        if new_name in output_tensor_names:
            print("[ERROR] The defined new_name '{}' is already exist in the onnx model, which is not allowed.")
            sys.exit(-1)

    for i, ipt in enumerate(model.graph.input):
        if ipt.name in args.original:
            idx = args.original.index(ipt.name)
            model.graph.input[i].name = args.new_name[idx]

    for i, node in enumerate(model.graph.node):
        for j, ipt in enumerate(node.input):
            if ipt in args.original:
                idx = args.original.index(ipt)
                model.graph.node[i].input[j] = args.new_name[idx]
        for j, out in enumerate(node.output):
            if out in args.original:
                idx = args.original.index(out)
                model.graph.node[i].output[j] = args.new_name[idx]

    for i, out in enumerate(model.graph.output):
        if out.name in args.original:
            idx = args.original.index(out.name)
            model.graph.output[i].name = args.new_name[idx]
    
    onnx.checker.check_model(model)
    onnx.save(model, args.save_dir)
    print("[Finished] The new model saved in {}.".format(args.save_dir))
    print("[DEBUG INFO] The inputs of new model: {}".format([x.name for x in model.graph.input]))
    print("[DEBUG INFO] The outputs of new model: {}".format([x.name for x in model.graph.output]))