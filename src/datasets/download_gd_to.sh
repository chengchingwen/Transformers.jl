#from https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
#1 should be the url, 2 should be the local dirname
# curl -sc /tmp/gcokie "${1}"
filename="$(curl -s -sc /tmp/gcokie "${1}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
curl -LOJb /tmp/gcokie "${1}&confirm=${getcode}"
echo "${2}/$filename"
