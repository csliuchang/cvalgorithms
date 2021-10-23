import os
import shutil
import cv2

format_list = ['JPG', 'png', 'PNG', 'bmp']


def assign_folder(data_root, keywords=" "):
    folder_path = os.path.join(data_root, 'labelme')
    template_path = os.path.join(data_root, 'template')
    folder_lists = [fold for fold in sorted(os.listdir(folder_path)) if fold.endswith('png')]
    template_folder_lists = sorted(os.listdir(template_path))
    save_path = os.path.join("/home/pupa/Datasets/Dock/labelme", 'assign_c4')
    assign_folder_path = template_path.replace('template', 'assign')
    if not os.path.exists(assign_folder_path):
        os.mkdir(assign_folder_path)
    assert len(template_folder_lists) == 1, 'template folder must not empty'
    mode = "one2many" if len(template_folder_lists) == 1 else "one2one"
    for i in range(len(folder_lists)):
        if folder_lists[i][-3:] in format_list:
            all_folder_path = os.path.join(folder_path, folder_lists[i])
            cur_folder_str = f"img_{keywords}_%03d." % i + folder_lists[i][-3:]
            template_path_str = f"img_{keywords}_%03d_template." % i + folder_lists[i][-3:]
            cur_all_folder_path = os.path.join(save_path, cur_folder_str)
            cur_template_all_folder_path = os.path.join(save_path, template_path_str)
            if mode == "one2many":
                all_template_all_folder_path = os.path.join(template_path, template_folder_lists[0])
            else:
                pass
            template_png = cv2.imread(all_template_all_folder_path, cv2.IMREAD_UNCHANGED)
            img_png = cv2.imread(all_folder_path, cv2.IMREAD_UNCHANGED)
            img_shape = img_png.shape[:-1]
            template_png = cv2.resize(template_png, (img_shape[1], img_shape[0]))
            # copy json path
            all_folder_label_path = os.path.join(folder_path, folder_lists[i].replace(folder_lists[i][-3:], 'json'))
            cur_folder_label_str = f"img_{keywords}_%03d." % i + 'json'
            cur_all_label_folder_path = os.path.join(save_path, cur_folder_label_str)
            cv2.imwrite(cur_template_all_folder_path, template_png)
            shutil.copy(all_folder_path, cur_all_folder_path)
            shutil.copy(all_folder_label_path, cur_all_label_folder_path)
            print(f"transport success and save in {cur_all_label_folder_path}")
    pass


def change_template(path, template_path):
    folders = os.listdir(path)
    keyword = 'glue00_roi0'
    for folder in folders:
        if keyword in folder and 'template' in folder:
            folder_str = os.path.join(path, folder)
            old_tem_fold = cv2.imread(folder_str, cv2.IMREAD_UNCHANGED)
            img_shape = old_tem_fold.shape[:-1]
            tem_fold = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
            tem_fold_resize = cv2.resize(tem_fold, (img_shape[1], img_shape[0]))
            cv2.imwrite(folder_str, tem_fold_resize)



if __name__ == '__main__':
    # print("you should put the format data as:"
    #       " data_root----labelme"
    #       "\n\t\t\t\t\t\t\t\t\t\t\t----template")
    # data_root = input("Input your data root path:")
    # keywords = input("Input your pic name:")
    #
    root = "/home/pupa/Datasets/Dock/labelme/dock_c4"
    data_root_list = os.listdir(root)
    for data_root_str in data_root_list:
        keywords = data_root_str
        data_root = os.path.join(root, data_root_str)
        # assign_path = os.path.join(data_root, 'assign')
        # shutil.rmtree(assign_path)
        assign_folder(data_root, keywords)