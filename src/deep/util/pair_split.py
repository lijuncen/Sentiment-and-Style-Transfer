# -*- coding: gb18030 -*-   
import codecs

def pair_split(input_file_path, output_file_path, charset="utf-8"):
    p_list = list()
    with  codecs.open(input_file_path, 'r', charset) as f:
        for line in f:
            line = line.strip()
            tokens = line.split('\t')
            if len(tokens) < 2:
                continue
            for token_q, token_a in zip(tokens[:-1], tokens[1:]):
                token_q = token_q.strip()
                token_a = token_a.strip()
                if len(token_q) == 0 or len(token_a) == 0:
                    continue
                if len(token_q) > 5 and len(token_a) > 5 and  (token_q[-1] == u'?' or token_q[-1] == u'？'):
                    print token_q, "\t", token_a
                    p_list.append((token_q, token_a))
    
    with codecs.open(output_file_path, 'w', charset) as f:
        for q, a in p_list:
            f.write("%s\t%s\n" % (q, a))
                
if __name__ == '__main__':
    pair_split("/home/huanyan/wx/dialog_small", "../generate_test_data_pair", charset='gb18030')
    
