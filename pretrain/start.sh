# cd jobs_masked_edge
# bash jobs.sh
# cd -

cd jobs_masked_node
bash jobs.sh
cd -

cd jobs_masked_node_edge
bash jobs.sh
cd -


cd jobs_masked_edge
bash jobs_large_tokenizer.sh
cd -

cd jobs_masked_node
bash jobs_large_tokenizer.sh
cd -

cd jobs_masked_node_edge
bash jobs_large_tokenizer.sh
cd -