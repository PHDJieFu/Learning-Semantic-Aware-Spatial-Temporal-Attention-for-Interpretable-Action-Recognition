
import os
 
f_dir = '/raid/fj/kinetics400/kinetics-400/'
f_list = os.listdir(f_dir)
f_basename = 'kinetics-400.tar.gz.part{}-{}'
print(len(f_list))
nn=0
dst_fname = '/raid/fj/kinetics400/kinetics-400-source.tar.gz'
dst_f = open(dst_fname,'wb')
for N in range(1,len(f_list)//4+1):
    for i in range(4):
        f_dstname = f_basename.format(N,i)
        print(f_dstname,os.path.exists(f_dir+f_dstname))
        src_f = open(f_dir+f_dstname,'rb')
        dst_f.write(src_f.read())
        dst_f.flush()
        src_f.close()
        nn+=1
dst_f.close()
print(nn)