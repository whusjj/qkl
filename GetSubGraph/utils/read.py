#  Copyright [2022] [MA WEI @ NTU], ma_wei@ntu.edu.sg

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#load parsers
from tree_sitter import Language, Parser
from utils.visit_ast import index_to_code_token, tree_to_token_index
import os
# print("------------------------------")
# print(os.getcwd())
lang="solidity"
parsers={}        
LANGUAGE = Language(f'{os.getcwd()}/utils/parser/my-languages.so', lang)
parser = Parser()
parser.set_language(LANGUAGE)   
parsers[lang]= parser

def split_functions(code):
    tree = parser.parse(bytes(code,'utf8'))  
    root_node = tree.root_node
    comments = []
    functions = []
    code = code.split("\n")
    for child_node in root_node.children:
        if child_node.type == "function_definition":
            fs = child_node.start_point[0]
            fe = child_node.end_point[0]
            if fs != fe:
                fcode = code[fs:fe+1]
            else:
                fcode = code[fs]
           # print(fcode)

    tokens_index = tree_to_token_index(root_node)
    code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
    #print(root_node.sexp())

    query = LANGUAGE.query("""
    (function_definition
      name: (identifier) @function.def)
    """)

    from pprint import pprint
    captures = query.captures(tree.root_node)
    #pprint(captures)
    res = {}
    for fdef in captures:
      #  print(type(fdef[0]))
        node=fdef[0].parent
       # print("=======")
        fncode = []
        for child_node in node.children:
            sp = child_node.start_point
            ep = child_node.end_point   
            fncode += [index_to_code_token((sp, ep), code)]
        # print(fncode)
        # print(" ".join(fncode) )
        res[fncode[1]]=" ".join(fncode)
    return res
    
def query_class(code):
    tree = parser.parse(bytes(code,'utf8'))  
    root_node = tree.root_node
    comments = []
    class_node = {}
    code = code.split("\n")
    for child_node in root_node.children:
       # print(child_node.type)
        if child_node.type == "contract_declaration":
            sp = child_node.start_point
            ep = child_node.end_point   
            ccode =  index_to_code_token((sp, ep), code) 
            class_node[ccode] =  child_node 
            # if 'FiatTokenProxy' in class_name:
            #     print(child_node.type)
            #     for n in child_node.children:
            #         if n.type == "contract_body":
            #             for nn in n.children:
            #                 if nn.type == "constructor_definition":
            #                     for nnn in nn.children:
            #                          print(nnn.type)
                    #print(n.type)
     
    query = LANGUAGE.query("""
     (contract_declaration
       name: (identifier)	@definition.class)
    """)
    
    # from pprint import pprint
    # captures = query.captures(class_node[0])
    #pprint(captures)
    res = {}
    for _, node in class_node.items():
        # print(fdef)
        # res.append(fdef)
        # print(type(fdef[0]))
        # node = fdef[0]
        captures = query.captures(node)[0]
       # print(captures)
        class_node = captures[0]
        class_name =  index_to_code_token((class_node.start_point, class_node.end_point), code) 
        fncs = list(keep_funcwSig(node, code).values())
        class_sig = []
        for cn in node.children:
            if cn.type != "contract_body":
                class_sig.append(  index_to_code_token((cn.start_point, cn.end_point), code)  )
            else:
                break
        constructor = []
        for cn in node.children:
            if cn.type == "contract_body":
                for nn in cn.children:
                   if nn.type == "constructor_definition":
                        for nnn in nn.children:
                            constructor.append(  index_to_code_token((nnn.start_point, nnn.end_point), code) )
        if len(constructor) != 0:
            fbody = " ".join(class_sig) + " { "+ " ".join(constructor) +" ; "+" ; ".join(fncs) + " } "
        else:
            fbody = " ".join(class_sig) + " { "+ " ; ".join(fncs) + " } "
        res[class_name] =  fbody 
    return res



def keep_funcwSig(root_node, code):
   

    query = LANGUAGE.query("""
    (function_definition
      name: (identifier) @function.def)
    """)
    
    from pprint import pprint
    captures = query.captures(root_node)
    #pprint(captures)
    res = {}
    for fdef in captures:
        # print(fdef)
        # res.append(fdef)
        # print(type(fdef[0]))
        node=fdef[0].parent
       # print("=======")
        #node=fdef[0]
        fncode = []
        for child_node in node.children:
            sp = child_node.start_point
            ep = child_node.end_point   
            if child_node.type != "function_body":
                fncode.append( index_to_code_token((sp, ep), code) )
        # print(fncode)
        # print(" ".join(fncode) )
        res[fncode[1]]=" ".join(fncode)
       # res.append( fncode )

    return res

