import json
import os
import sys
from six.moves import urllib
import scipy.io


#get synset_to_human dict
print('Loading synset id to human readable class name dictionary...')
base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url) 
filename, _ = urllib.request.urlretrieve(synset_to_human_url)                             
synset_to_human_list = open(filename).readlines()

synset_to_human = {}
for s in synset_to_human_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human
    
#get ground truth
print('Loading ground truth .mat file...')
meta_data = scipy.io.loadmat('ILSVRC2014_devkit/data/meta_clsloc.mat')

wnid_to_label = {m[1][0]: m[0][0][0] for m in meta_data['synsets'][0]}
label_to_wnid = {m[0][0][0]: m[1][0] for m in meta_data['synsets'][0]}
label_names_dict = {m[1][0]: m[2][0] for m in meta_data['synsets'][0]}

print('Loading validation ground truth...') 
gt_data = {}
with open('ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt') as f:
    for i,line in enumerate(f):
        gt_data[i+1] = label_to_wnid[int(line.strip())]
    
#get images and classes
print('Converting data to COCO format...')
base_folder = '/home/ubuntu/imagenet_data/'
split = 'val'
folder = base_folder + split + '/'
files = os.listdir(folder)

images = []                       
annotations = []                           
cats = []     
ann_id_count = 0
for file in files:
    im_num = int(file.split('_')[-1].replace('.JPEG',''))
    cat_id = gt_data[im_num]
    cats.append(cat_id)                    
    im = {}   
    im['file_name'] = folder+'/'+file
    im['id'] = file.split('.')[0]
    images.append(im)
    ann = {}
    ann['image_id'] = im['id']
    ann['id'] = str(ann_id_count)
    ann_id_count += 1
    ann['category_id'] = cat_id
    annotations.append(ann)
            
print(len(images),len(annotations),len(cats))

cats = list(set(cats))
categories = []
for cat_id in cats:
    cat = {}
    cat['id'] = cat_id
    cat['name'] = label_names_dict[cat_id]
    categories.append(cat)
    
data = {}
data['images'] = images
data['categories'] = categories
data['annotations'] = annotations
      
print('Saving COCO validation database...')      
    
json.dump(data,open('ILSVRC2014/ILSVRC2014_'+split+'_classification.json','w'))
    
    