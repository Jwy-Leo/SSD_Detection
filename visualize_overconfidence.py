import numpy as np
import cv2
def main():
    image_index = 0
    histogram = dataset_category_statastic(image_index)
    main_feature_images = Main_feature_finding()
    correlation_matrix = feature_dependency(main_feature_images)
    import pdb;pdb.set_trace()
def dataset_category_statastic(image_index, categories = 10):
    def simulate_fetch(index, categories):
        sample_number = np.random.choice(20)
        sample_list = np.random.choice(categories,sample_number)
        return sample_list
    sample_list = simulate_fetch(image_index,categories)
    category_item, reference_table = np.histogram(sample_list, categories, (0,categories))
    histogram = category_item
    return histogram
def Main_feature_finding(categories = 10):
    def simulate_category_sample(image_size = (640,480,3)):
        sample_w, sample_h = int(np.random.rand() * image_size[0]), int(np.random.rand() * image_size[1])
        sim_image = np.random.rand(sample_w, sample_h, image_size[2])
        return sim_image
    def synthtic_table():
        sample_number = np.random.choice(300)
        sample_list = np.random.choice(categories,sample_number)
        category_item, reference_table = np.histogram(sample_list, categories, (0,categories))
        return category_item
    def find_feature(image_list):
        data = np.stack(image_list)
        flatten_features = data.reshape(data.shape[0],-1)
        mean_feature = flatten_features.mean(0)
        sim_values = np.matmul(flatten_features, mean_feature)
        sample_index = np.argmax(sim_values)
        return image_list[sample_index]
        
    categorys = synthtic_table()
    main_feature_image = []
    for _category_number_item in categorys:
        image_list = [cv2.resize(simulate_category_sample(),(256,256)) for i in range(_category_number_item)]
        mean_image = find_feature(image_list)
        main_feature_image.append(mean_image)
    return main_feature_image

def feature_dependency(main_feature_images):
    data = np.stack(main_feature_images)
    flatten_features = data.reshape(data.shape[0],-1)
    correlation = np.matmul(flatten_features,flatten_features.transpose())
    import pdb;pdb.set_trace()ˊ
    return correlation / float(flatten_features.shape[1])

if __name__ =="__main__":
    main()
    