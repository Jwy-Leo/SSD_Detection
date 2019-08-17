import yaml
import argparse
import os 
# Feature abalation study
def arguments():
    parse=argparse.ArgumentParser(description="Mnist Classification Environment")
    parse.add_argument("--filename",type=str,default="config/test",help="configuration")
    #parse.add_argument("--write_config",action="store_true",help="write configuration")
    parse.add_argument("--read_config",action="store_true",help="read configuration")
    # Generate the folder if it isn't exists
    args = parse.parse_args()
    folder = "/".join(str.split(args.filename,"/")[:-1])
    
    if (folder is not "") and not os.path.exists(folder):
        os.makedirs(folder)

    print(args)
    
    return args
def main(args):
    #if args.write_config:
    #    save_model_configuration(args.filename)
    if args.read_config:
        if args.filename[-5:]!=".yaml":
            args.filename+=".yaml"
        if not os.path.exists(args.filename):
            raise ValueError("Doesn't exist this file")
        load_model_configuration(args.filename)
    
def load_model_configuration(name):
    Code_usuage_help = \
    " \
    -------------------------------------------------------\n \
    The system have three compount : \n \
    1. Dataset setting\n \
    2. Active model(Classification model) training hyper parameters\n \
    3. data preprocess\n \
    4. detection model setting\n \
    -------------------------------------------------------\n \
    "
    print(Code_usuage_help)
    with open(name,"r") as F:
        data_loaded = yaml.load(F)
    def read_dict(data):
        assert isinstance(data,dict)
        key_table = data.keys()
        print("{}".format(key_table))
        for key in key_table:
            if isinstance(data[key],dict):
                print("%s:"%key)
                read_dict(data[key])
            else:
                print("{0:<20s}:\t{1:}".format(key,data[key]))
    assert isinstance(data_loaded,dict), "configuration is not dict type, we save it as *.yaml file"
    print("{} configuration load".format(name))
    read_dict(data_loaded)
    return data_loaded
if __name__=="__main__":
    args = arguments()
    main(args)
