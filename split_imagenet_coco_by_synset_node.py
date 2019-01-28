import json
import urllib

class_wnid = "n00015388"
class_name = "animal"

print('Getting full list of leaves under synset node '+class_wnid+'...')

urllib.urlretrieve("http://image-net.org/api/text/wordnet.structure.hyponym?wnid="+class_wnid+"&full=1", class_name+"_synsets.txt") 

classes = []                                              
with open(class_name+'_synsets.txt','r') as f:                 
    for line in f:             
        classes.append(line.strip().replace('-',''))
        
print('Loading the data to split...')

data_to_split = 'ILSVRC2014/ILSVRC2014_train_classification.json'
data = json.load(open(data_to_split,'r'))

#remove locally specific path if necessary
for im in data['images']:                                     
    im['file_name'] = im['file_name'].replace('/home/ubuntu/imagenet_data/','') 

print('Splitting data by class...')

class_anns = []
non_class_anns = []

for ann in data['annotations']:                           
    if ann['category_id'] in classes:                              
        class_anns.append(ann)
    else:                                           
        non_class_anns.append(ann)
        
im_id_to_im = {im['id']:im for im in data['images']} 
class_ims = [im_id_to_im[ann['image_id']] for ann in class_anns]
non_class_ims = [im_id_to_im[ann['image_id']] for ann in non_class_anns]

class_cats = [cat for cat in data['categories'] if cat['id'] not in classes]
non_class_cats = [cat for cat in data['categories'] if cat['id'] not in classes]

class_data = {'images':class_ims,'categories':class_cats,'annotations':class_anns}
non_class_data = {'images':non_class_ims,'categories':non_class_cats,'annotations':non_class_anns}

print('Saving data to .json...')

class_output_file = data_to_split.replace('.json','')+'_'+class_name+'.json'
json.dump(class_data,open(class_output_file,'w'))

non_class_output_file = data_to_split.replace('.json','')+'_non_'+class_name+'.json'
json.dump(non_class_data,open(non_class_output_file,'w'))