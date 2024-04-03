import re

def get_index_list(str):
    split_list = ["from:", "to:", "call:", "params:", "returns:"]
    index_list = []
    for split_item in split_list:
        index_list.append(str.find(split_item))
    return index_list


def get_next(i, index_list,str):
    for index in range(i+1, len(index_list)):
        if index_list[index]!= -1:
            return index_list[index]
    return len(str)


def get_dict(str, index_list):
    split_list = ["from:", "to:", "call:", "params:", "returns:"]
    split_list_pro = [item[:-1] for item in split_list]
    my_dict = {}
    for i in range(5):
        if index_list[i]!=-1:
            my_dict[split_list_pro[i]] = (str[index_list[i]+len(split_list[i]): get_next(i, index_list,str)]).strip()
        else:
            my_dict[split_list_pro[i]] = ""
    return my_dict

def process_one_trace(str):
    split = str.lstrip("-")
    index_list = get_index_list(split)
    dict = get_dict(split, index_list)
    return dict


def trace_to_dict(trace):
    split_trace = trace.split("\n")
    trace_dict = []
    index_call = 0
    for tra in split_trace[:-1]:
        trace_dict.append(process_one_trace(tra))
    return trace_dict

def get_node_trace_info(trace_dict, trace):
    node_list = []
    for i in range(len(trace_dict)):
        if trace_dict[i]["from"] not in node_list:
            node_list.append(trace_dict[i]["from"])
        if trace_dict[i]["to"] not in node_list:
            node_list.append(trace_dict[i]["to"])

    for i, item in enumerate(node_list):
        if item in trace:
            trace = trace.replace(item, str(i))
    return trace, node_list

# trace_dict = trace_to_dict(trace)
# trace_info , node_list = get_node_trace_info(trace_dict, trace)
# for addr in node_list[2:]:
#     print(addr)

# trace = 'from: 0xb1ca82a1e6a6255bc66b9330b08b642b07419469\tto: 0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\tcall: 0x7efc9f7d000000000000000000000000965fd3146f44ab652104a1dca94c8988de245cf9000000000000000000000000197ef010f6d11a1af6edd55c67cb85417a6d8b83\n-from: 0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\tto: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tcall: swapExactETHForTokens\tparams:1685893031,[0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2,0x197ef010f6d11a1af6edd55c67cb85417a6d8b83],0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\treturns: [100000000000000000,876188769059044808438]\n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tcall: getReserves\tparams:\treturns: 13235677930667936\n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: deposit\tparams:\treturns: \n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: transfer\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0,100000000000000000\treturns: True\n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tcall: swap\tparams:0,0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8,bytes\treturns: \n---from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0x197ef010f6d11a1af6edd55c67cb85417a6d8b83\tcall: transfer\tparams:0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8,876188769059044808438\treturns: True\n---from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0x197ef010f6d11a1af6edd55c67cb85417a6d8b83\tcall: balanceOf\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\treturns: 116318478974261828081\n---from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: balanceOf\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\treturns: 113235677930667936\n-from: 0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: balanceOf\tparams:0x965fd3146f44ab652104a1dca94c8988de245cf9\treturns: 4010000000000000000\n-from: 0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\tto: 0x965fd3146f44ab652104a1dca94c8988de245cf9\tcall: 0xc57f1922000000000000000000000000197ef010f6d11a1af6edd55c67cb85417a6d8b8300000000000000000000000000000000000000000000000621b20a594b97440000000000000000000000000000000000000000000000000037a661c10d510000\n--from: 0x965fd3146f44ab652104a1dca94c8988de245cf9\tto: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tcall: swapTokensForExactTokens\tparams:1685893031,[0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2,0x197ef010f6d11a1af6edd55c67cb85417a6d8b83],0x965fd3146f44ab652104a1dca94c8988de245cf9\treturns: [4002010792149011842,113108478970000000000]\n---from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tcall: getReserves\tparams:\treturns: 113235677930667936\n---from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: transferFrom\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0,4002010792149011842\treturns: \n---from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tcall: swap\tparams:0,0x965fd3146f44ab652104a1dca94c8988de245cf9,bytes\treturns: \n----from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0x197ef010f6d11a1af6edd55c67cb85417a6d8b83\tcall: transfer\tparams:0x965fd3146f44ab652104a1dca94c8988de245cf9,113108478970000000000\treturns: True\n----from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0x197ef010f6d11a1af6edd55c67cb85417a6d8b83\tcall: balanceOf\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\treturns: 3210000004261828081\n----from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: balanceOf\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\treturns: 4115246470079679778\n-from: 0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\tto: 0x197ef010f6d11a1af6edd55c67cb85417a6d8b83\tcall: approve\tparams:0x7a250d5630b4cf539739df2c5dacb4c659f2488d,115792089237316195423570985008687907853269984665640564039457584007913129639935\treturns: \n-from: 0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\tto: 0x197ef010f6d11a1af6edd55c67cb85417a6d8b83\tcall: balanceOf\tparams:0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\treturns: 876188769059044808438\n-from: 0x7ab322d2104e82bdf2ba32851df31f4f91ddbaf8\tto: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tcall: swapExactTokensForETH\tparams:1685893031,[0x197ef010f6d11a1af6edd55c67cb85417a6d8b83,0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2],0xb1ca82a1e6a6255bc66b9330b08b642b07419469\treturns: [876188769059044808438,4100179875497691522]\n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tcall: getReserves\tparams:\treturns: 4115246470079679778\n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0x197ef010f6d11a1af6edd55c67cb85417a6d8b83\tcall: transferFrom\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0,876188769059044808438\treturns: \n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tcall: swap\tparams:4100179875497691522,0x7a250d5630b4cf539739df2c5dacb4c659f2488d,bytes\treturns: \n---from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: transfer\tparams:0x7a250d5630b4cf539739df2c5dacb4c659f2488d,4100179875497691522\treturns: True\n---from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0x197ef010f6d11a1af6edd55c67cb85417a6d8b83\tcall: balanceOf\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\treturns: 879398769063306636519\n---from: 0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: balanceOf\tparams:0xde5e5f7b4a041a04653b1a1478d757efbd1a6ef0\treturns: 15066594581988256\n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tcall: withdraw\tparams:4100179875497691522\treturns: \n---from: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2\tto: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tcall: send ether(4.10018)\n--from: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d\tto: 0xb1ca82a1e6a6255bc66b9330b08b642b07419469\tcall: send ether(4.10018)\n'
#
# trace_dict = trace_to_dict(trace)
# trace_info , node_list = get_node_trace_info(trace_dict, trace)
# print(trace_info)
# # 分割字符串为行，并处理每一行
# new_text = "\n".join([line.split("params")[0] for line in trace_info.split("\n")])
#
# print(new_text)




