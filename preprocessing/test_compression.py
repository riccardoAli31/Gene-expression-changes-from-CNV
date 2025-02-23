name='ENSG00000090104'
emb = torch.load('ENSG00000090104.pt')
from scipy.sparse import sparray, save_npz
emb_s = emb.to_sparse()
torch.save(emb_s, name + '_sparse.pt')

mmwrite(name + '.mtx', emb)

with open(name + '.pkl', 'bw') as out_file:
    pickle.dump(emb, out_file)

with open(name + '_sparse.pkl', 'bw') as out_file:
    pickle.dump(emb_s, out_file)

# TODO: 
# * scipy.coo_array: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html#scipy.sparse.coo_array
# * scipy.mmwrite: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html
# * pickle.dump: https://docs.python.org/3/library/pickle.html#pickle.dump

#/vol/storage/shared/cmscb8/data/embeddings/batch_2/single_gene_barcode/AAACCAACATTGCGGT-2$ ll -h ENSG00000090104*
# -rw-rw-r-- 1 marw student 5.7K Feb 21 13:18 ENSG00000090104.mtx.gz
# -rw-rw-r-- 1 marw student 548K Feb 21 13:16 ENSG00000090104.pkl
# -rw-rw-r-- 1 marw student  11K Feb 21 12:47 ENSG00000090104.pt.gz
# -rw-rw-r-- 1 marw student 236K Feb 21 13:16 ENSG00000090104_sparse.pkl
# -rw-rw-r-- 1 marw student 237K Feb 21 13:07 ENSG00000090104_sparse.pt