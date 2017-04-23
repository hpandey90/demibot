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

urlretrieve('https://www.dropbox.com/s/2w2bsytkg4cib6y/dataset.zip?dl=1', 'data.zip', reporthook)

print("Unzipping Data")
zip_ref = zipfile.ZipFile('data.zip', 'r')
zip_ref.extractall()
zip_ref.close()
