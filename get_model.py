from urllib.request import urlretrieve
import zipfile
import sys
print("Downloading Data")
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))

urlretrieve('https://www.dropbox.com/s/w09jb8lvxeefz38/model.zip?dl=1', 'ckpt.zip', reporthook)

print("Unzipping Data")
zip_ref = zipfile.ZipFile('ckpt.zip', 'r')
zip_ref.extractall()
zip_ref.close()
