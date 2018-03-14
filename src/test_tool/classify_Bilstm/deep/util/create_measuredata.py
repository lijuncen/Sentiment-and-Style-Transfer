import codecs

def create_measure_data():
    input_file_path = "../data/measure/origin"
    input_file = codecs.open(input_file_path, "r", "utf-8", "ignore")
    
    label_index = 0
    label_dict = dict()
    
    output_file = codecs.open("../data/measure/base", "w", "utf-8")
    
    for line in input_file:
        line = line.strip("\n")
#         print line
        tokens = line.split("\t")
        question = tokens[0]
        label_text = tokens[1]
        
        if("#" in label_text):
            label_text = label_text.split("#")[0]
        
        label = label_dict.get(label_text)
        if(label is None):
            label = label_index
            label_dict[label_text] = label_index
            print label_text
            label_index = label_index + 1
        
        output_file.write(question)
        output_file.write("\t")
        output_file.write(str(label))
        output_file.write("\n")

    output_file.close()
    print "label_index: ", label_index
        
if __name__ == '__main__':
    print "Started!"
    create_measure_data()
    print "All finished!"
        