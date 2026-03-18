import os
import shutil
import random
import json
import pdb

# CLASS = ['Ford_class_aircraft_carriers', 'Arleigh_Burke_Class_Destroyer', 'Udaloy_Class_Destroyer',
#                'Sovremenny_Class_Destroyer', 'Oliver_Hazard_Perry_Class_Frigate', 'Ticonderoga_Class_Cruiser', 'Wasp_Class_Amphibious_Assault_Ship',
#                'Whidbey_Island_Class', 'SanAntonio_Class_Amphibious_Transport_Dock', 'A_submarine', 'Submarine',
#          'Missile_Boat', 'Nimitz_class_aircraft_carrier', 'ship']

CLASS = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter']

add_classes = ['福特级', '伯克级',
'佩里级',
'提康德罗加级',
'黄蜂级',
'惠德贝岛级',
'圣安东尼奥级',
'尼米兹级']

trans = {'福特级':'Ford_class_aircraft_carriers', '伯克级':'Arleigh_Burke_Class_Destroyer',
'佩里级':'Oliver_Hazard_Perry_Class_Frigate',
'提康德罗加级':'Ticonderoga_Class_Cruiser',
'黄蜂级':'Wasp_Class_Amphibious_Assault_Ship',
'惠德贝岛级':'Whidbey_Island_Class',
'圣安东尼奥级':'SanAntonio_Class_Amphibious_Transport_Dock',
'尼米兹级':'Nimitz_class_aircraft_carrier'}

def split_tif_txt(path, img_path, txt_path):
    items = os.listdir(path)
    imgs = []
    txts = []
    for item in items:
        #pdb.set_trace()
        if item[-4:] == ".txt":
            if item == "classes.txt":
                continue
            else:
                txts.append(os.path.join(path, item))
        elif item[-4:] == ".tif":
            imgs.append(os.path.join(path, item))
        else:
            continue
    imgs.sort()
    txts.sort()
    print(len(imgs))
    if len(imgs) == len(txts):
        total = 0
        for i in range(len(imgs)):
            src_img = imgs[i]
            src_txt = txts[i]
            dest_img = os.path.join(img_path, '{:0>4}'.format(i)+".tif")
            dest_txt = os.path.join(txt_path, '{:0>4}'.format(i)+".txt")
            if src_txt[:-4] == src_txt[:-4]:
                shutil.copy(src_txt, dest_txt)
                shutil.copy(src_img, dest_img)
                total = total + 1
            else:
                continue
        print(total)
    else:
        print("error")

def ins_num(txt_path):
    # classes = {'Ford_class_aircraft_carriers': 0, 'Arleigh_Burke_Class_Destroyer': 0, 'Udaloy_Class_Destroyer': 0,
    #            'Sovremenny_Class_Destroyer': 0, 'Oliver_Hazard_Perry_Class_Frigate': 0, 'Ticonderoga_Class_Cruiser': 0,
    #            'Wasp_Class_Amphibious_Assault_Ship': 0,
    #            'Whidbey_Island_Class': 0, 'SanAntonio_Class_Amphibious_Transport_Dock': 0, 'A_submarine': 0,
    #            'Submarine': 0, 'Missile_Boat': 0, 'Nimitz_class_aircraft_carrier': 0, 'ship': 0}
    #
    # class_txt = {'Ford_class_aircraft_carriers': [], 'Arleigh_Burke_Class_Destroyer': [], 'Udaloy_Class_Destroyer': [],
    #              'Sovremenny_Class_Destroyer': [], 'Oliver_Hazard_Perry_Class_Frigate': [],
    #              'Ticonderoga_Class_Cruiser': [], 'Wasp_Class_Amphibious_Assault_Ship': [],
    #              'Whidbey_Island_Class': [], 'SanAntonio_Class_Amphibious_Transport_Dock': [], 'A_submarine': [],
    #              'Submarine': [], 'Missile_Boat': [], 'Nimitz_class_aircraft_carrier': [], 'ship': []}
    classes = {'plane':0, 'baseball-diamond':0, 'bridge':0, 'ground-track-field':0,
                'small-vehicle':0, 'large-vehicle':0, 'ship':0, 'tennis-court':0,
                'basketball-court':0, 'storage-tank':0, 'soccer-ball-field':0,
                'roundabout':0, 'harbor':0, 'swimming-pool':0, 'helicopter':0}

    class_txt = {'plane':[], 'baseball-diamond':[], 'bridge':[], 'ground-track-field':[],
                'small-vehicle':[], 'large-vehicle':[], 'ship':[], 'tennis-court':[],
                'basketball-court':[], 'storage-tank':[], 'soccer-ball-field':[],
                'roundabout':[], 'harbor':[], 'swimming-pool':[], 'helicopter':[]}

    txts = os.listdir(txt_path)
    for txt in txts:
        name = os.path.join(txt_path, txt)
        with open(name, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                for line in lines:
                    context = line.split(" ")
                    if len(context) < 5:
                        continue
                    if len(context) > 9 and not context[9] == '2\n' or len(context) <= 9:
                        if len(context) > 9:
                            classes[context[8]] = classes[context[8]] + 1
                            class_txt[context[8]].append(name)
                        else:
                            classes[context[8].split('\n')[0]] = classes[context[8].split('\n')[0]] + 1
                            class_txt[context[8].split('\n')[0]].append(name)
        f.close()
    for i in range(len(CLASS)):
        orig_list = class_txt[CLASS[i]]
        class_txt[CLASS[i]] = list(set(orig_list))
        class_txt[CLASS[i]].sort()
    print(classes)
    #print(class_txt)
    for i in range(len(CLASS)):
        print("{}, {}".format(CLASS[i],len(class_txt[CLASS[i]])))

    return class_txt

def gen_test(class_txt):

    schedule = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 5]
    test = []
    for i in range(len(CLASS)):
        l = random.sample(class_txt[CLASS[i]], schedule[i])
        for j in range(len(l)):
            test.append(l[j])
    orig_list = test
    test = list(set(orig_list))
    print(test)
    print(len(orig_list))
    print(len(test))
    names = []
    for i in range(len(test)):
        names.append(os.path.basename(test[i])[:-4])

    return names

def div_train_val(img_path, txt_path, test, out_path):
    train_path = os.path.join(out_path, "train")
    train_label_path = os.path.join(out_path, "train_labels")
    val_path = os.path.join(out_path, "val")
    val_label_path = os.path.join(out_path, "val_labels")

    # shutil.rmtree(train_path)
    # shutil.rmtree(train_label_path)
    # shutil.rmtree(val_path)
    # shutil.rmtree(val_label_path)

    os.mkdir(train_path)
    os.mkdir(train_label_path)
    os.mkdir(val_path)
    os.mkdir(val_label_path)

    imgs = os.listdir(img_path)
    for img in imgs:
        name = os.path.join(img_path, img)
        if img[:-4] in test:
            shutil.copy(name, os.path.join(val_path, img))
        else:
            shutil.copy(name, os.path.join(train_path, img))
    txts = os.listdir(txt_path)
    for txt in txts:
        name = os.path.join(txt_path, txt)
        if txt[:-4] in test:
            shutil.copy(name, os.path.join(val_label_path, txt))
        else:
            shutil.copy(name, os.path.join(train_label_path, txt))


def div_train_val_rand(img_path, txt_path, out_path):
    train_path = os.path.join(out_path, "train")
    train_label_path = os.path.join(out_path, "train_labels")
    val_path = os.path.join(out_path, "val")
    val_label_path = os.path.join(out_path, "val_labels")

    shutil.rmtree(train_path)
    shutil.rmtree(train_label_path)
    shutil.rmtree(val_path)
    shutil.rmtree(val_label_path)

    os.mkdir(train_path)
    os.mkdir(train_label_path)
    os.mkdir(val_path)
    os.mkdir(val_label_path)

    imgs = os.listdir(img_path)
    val = random.sample(imgs, int(len(imgs)*0.1))

    for img in imgs:
        if img in val:
            src_img = os.path.join(img_path, img)
            dest_img = os.path.join(val_path, img)
            src_txt = os.path.join(txt_path, img[:-4]+'.txt')
            dest_txt = os.path.join(val_label_path, img[:-4]+'.txt')
            shutil.copy(src_txt, dest_txt)
            shutil.copy(src_img, dest_img)
        else:
            src_img = os.path.join(img_path, img)
            dest_img = os.path.join(train_path, img)
            src_txt = os.path.join(txt_path, img[:-4] + '.txt')
            dest_txt = os.path.join(train_label_path, img[:-4] + '.txt')
            shutil.copy(src_txt, dest_txt)
            shutil.copy(src_img, dest_img)



def remove_txt_jpg(img_dir, txt_dir):
    imgs = os.listdir(img_dir)
    txts = os.listdir(txt_dir)
    imgs.sort()
    txts.sort()

    rms = []

    if len(imgs) == len(txts):
        print(len(txts))
        for txt in txts:
            name = os.path.join(txt_dir, txt)
            with open(name, 'r') as f:
                lines = f.readlines()
                if len(lines) < 1:
                    rms.append(txt[:-4])
            f.close()
        print(len(rms))

        for i in range(len(rms)):
            #print(rms[i])
            rm_img = os.path.join(img_dir, rms[i]+'.png')
            rm_txt = os.path.join(txt_dir, rms[i]+'.txt')
            os.remove(rm_img)
            os.remove(rm_txt)
            #print(rm_txt+" "+rm_img)
    else:
        print("error")

def rename_txt_png(img_path, txt_path):
    imgs = os.listdir(img_path)
    imgs.sort()
    txts = os.listdir(txt_path)
    txts.sort()

    if len(imgs) == len(txts):
        for i in range(len(imgs)):
            name = '{:0>6}'.format(i)
            if imgs[i][:-4] == txts[i][:-4]:
                src_img = os.path.join(img_path, imgs[i])
                dest_img = os.path.join(img_path, name+'.png')
                src_txt = os.path.join(txt_path, txts[i])
                dest_txt = os.path.join(txt_path, name+'.txt')

                os.rename(src_txt, dest_txt)
                os.rename(src_img, dest_img)
            else:
                continue
    else:
        print("error")

def gen_sup(percent, train_path, sample_num):

    sup_dict = {}
    seed_0 = {}
    txts = os.listdir(train_path)
    class_txt = ins_num(train_path)

    select_samples = int(len(txts) * (percent / 100))
    print(select_samples)

    ins = []

    for i in range(len(CLASS)):
        if len(class_txt[CLASS[i]]) >= sample_num:
            ins = ins + random.sample(class_txt[CLASS[i]], sample_num)
        else:
            ins = ins + class_txt[CLASS[i]]

    orignal_list = ins
    ins = list(set(orignal_list))
    rest = select_samples - len(ins)
    print("rest: {}".format(rest))
    #print(len(orignal_list))
    #print(len(ins))

    flag = 0
    for i in range(len(txts)):
        if flag >= rest:
            break
        else:
            if txts[i] not in ins:
                ins.append(txts[i])
                flag = flag + 1
            else:
                continue
    print(len(ins))

    for i in range(len(ins)):
        ins[i] = int(os.path.basename(ins[i][:-4]))
    print(ins)

    ins_json = []
    for i in range(len(ins)):
        ins_json.append(str(ins[i]).zfill(6)+'.png')

    # seed_0["0"] = ins
    # sup_dict[str(percent)] = seed_0
    # #print(sup_dict)
    #
    with open('D:/SOOD/data_lists/ships_{}_supervision.json'.format(int(percent)), 'w') as f:
        f.write(json.dumps(ins_json))

def combine_apr_ships(imgPath, txtPath, outPath):
    apr_list = os.listdir(txtPath)
    add_img = []
    add_txt = []
    for txt in apr_list:
        with open(os.path.join(txtPath, txt), 'r') as f:
            lines = f.readlines()
            flag = 0
            for line in lines:
                items = line.split(' ')
                if items[0] in add_classes:
                    flag = 1
                    break
                else:
                    continue
            f.close()
        if flag == 1:
            add_txt.append(os.path.join(txtPath, txt))
            add_img.append(os.path.join(imgPath, txt[:-4]+'.jpg'))
        else:
            continue

    add_img.sort()
    add_txt.sort()
    assert len(add_txt) == len(add_img), 'error'
    for i in range(len(add_txt)):
        img_src = add_img[i]
        img_dest = os.path.join(os.path.join(outPath, 'train'), '{:0>6}'.format(i)+'.png')
        txt_src = add_txt[i]
        txt_dest = os.path.join(os.path.join(outPath, 'train_labels'), '{:0>6}'.format(i)+'.txt')
        assert os.path.basename(img_src)[:-4] == os.path.basename(txt_src)[:-4], 'error'
        shutil.copy(img_src, img_dest)
        shutil.copy(txt_src,txt_dest)
        #print(os.path.basename(txt_dest))

    dest_txts = os.listdir(os.path.join(os.path.join(outPath, 'train_labels')))
    dest_txts.sort()
    for i in range(len(dest_txts)):
        txt = dest_txts[i]
        print(txt)
        with open(os.path.join(os.path.join(os.path.join(outPath, 'train_labels')), txt), 'r') as f:
            save_lines = []
            lines = f.readlines()
            for line in lines:
                cls = line.split(' ')[0]
                if cls in add_classes:
                    save_lines.append(line)
            f.close()
        with open(os.path.join(os.path.join(os.path.join(outPath, 'train_labels')), txt), 'w') as f:
            for save_line in save_lines:
                llist = save_line.split(' ')
                #pdb.set_trace()
                str = llist[1] + ' ' + llist[2] + ' ' + llist[3] + ' ' + llist[4] + ' ' + llist[5] + ' ' +\
                 llist[6] + ' ' + llist[7] + ' ' + llist[8][:-1] + ' ' + trans[llist[0]] + ' ' + '0\n'
                f.write(str)
            f.close()

def mv_background(src_path, dest_path):
    images = os.listdir(os.path.join(src_path, 'train'))
    txts = os.listdir(os.path.join(src_path, 'train_labels'))

    for txt in txts:
        with open(os.path.join(os.path.join(src_path, 'train_labels'), txt), 'r') as f:
            lines = f.readlines()
            if len(lines) < 1:
                f.close()
                src_img = os.path.join(os.path.join(src_path, 'train'), txt[:-4]+'.png')
                src_txt = os.path.join(os.path.join(src_path, 'train_labels'), txt)
                dest_img = os.path.join(dest_path, 'images')
                dest_txt = os.path.join(dest_path, 'labels')
                shutil.move(src_img, dest_img)
                shutil.move(src_txt, dest_txt)

def merge_f2b_origin(src_img_dir, src_txt_dir, f2b_img_dir, f2b_txt_dir):
    f2b_img = (os.listdir(f2b_img_dir))
    f2b_txt = (os.listdir(f2b_txt_dir))
    f2b_img.sort()
    f2b_txt.sort()

    assert len(f2b_img) == len(f2b_txt)
    print(len(f2b_img))

    for i in range(len(f2b_img)):
        src1 = os.path.join(f2b_img_dir, f2b_img[i])
        dest1 = os.path.join(src_img_dir, str(int(f2b_img[i][:-4])+6166).zfill(6)+'.png')
        src2 = os.path.join(f2b_txt_dir, f2b_txt[i])
        dest2 = os.path.join(src_txt_dir, str(int(f2b_txt[i][:-4])+6166).zfill(6) + '.txt')
        print(dest1, dest2)
        shutil.copy(src1, dest1)
        shutil.copy(src2, dest2)









if __name__ == "__main__":
    dir = r"D:/data/ships/total"
    img_path = "D:/data/ships/images"
    txt_path = "D:/data/ships/labels"

    #merge_f2b_origin('/media/disk2/wyh/ships_origin_1024/train/', '/media/disk2/wyh/ships_origin_1024/train_labels/',
    #                 '/media/disk2/wyh/fore2back/images/', '/media/disk2/wyh/fore2back/labels/')
    #combine_apr_ships("/media/disk2/wyh/APR_F_coco/images/", "/media/disk2/wyh/APR_F_coco/labels/", "/media/disk2/wyh/ships_AddShips/")
    #mv_background('/media/disk2/wyh/ships_origin_1024/', '/media/disk2/wyh/ships_background/')
    #div_train_val_rand(img_path, txt_path, '/media/disk2/wyh/ships/')
    class_txt = ins_num(r'D:\data\DOTA_v1.0\split_ss_dota\train_ori\annfiles')
    #names = gen_test(class_txt)
    #div_train_val(img_path, txt_path, names, 'D:/data/ships/')
    #class_txt = ins_num('/media/disk2/wyh/ships_noship/train_labels/')
    #print(class_txt['Ford_class_aircraft_carriers'])
    #print(class_txt['Arleigh_Burke_Class_Destroyer'])
    #print(class_txt['Submarine'])

    #split_tif_txt(dir, img_path, txt_path)
    #remove_txt_jpg(r'D:\data\DOTA_v1.0\split_ss_dota\val\images', r'D:\data\DOTA_v1.0\split_ss_dota\val\annfiles')
    #ins_num('/media/disk2/wyh/ships/train_labels/')
    #rename_txt_png('/media/disk2/wyh/ships_origin_1024/train/', '/media/disk2/wyh/ships_origin_1024/train_labels/')
    #gen_sup(40.0, r'D:\data\ships_withAug_1024\train_labels', 500)


