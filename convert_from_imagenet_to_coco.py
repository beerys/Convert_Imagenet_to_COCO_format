import json
import os
import sys
from six.moves import urllib


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
    
#get images and classes (training only)
print('Reading ground truth from file structure and creating COCO database...')
base_folder = '/home/ubuntu/imagenet_data/'
split = 'train'
folder = base_folder + split + '/'
files = os.listdir(folder)

images = []                       
annotations = []                           
cats = []     
ann_id_count = 0
for directory in files:
    if os.path.isdir(folder+directory):
        cat_id = directory                     
        cats.append(cat_id)                    
        path = os.listdir(folder+directory+'/')
        for filename in path:
            im = {}   
            im['file_name'] = folder+directory+'/'+filename
            im['id'] = filename.split('.')[0]
            images.append(im)
            ann = {}
            ann['image_id'] = im['id']
            ann['id'] = str(ann_id_count)
            ann_id_count += 1
            ann['category_id'] = cat_id
            annotations.append(ann)
            
print(len(images),len(annotations),len(cats))

categories = []
for cat_id in cats:
    cat = {}
    cat['id'] = cat_id
    cat['name'] = synset_to_human[cat_id]
    categories.append(cat)
    
data = {}
data['images'] = images
data['categories'] = categories
data['annotations'] = annotations

print('Saving COCO training database...')
json.dump(data,open('ILSVRC2014/ILSVRC2014_'+split+'_classification.json','w'))
    
    