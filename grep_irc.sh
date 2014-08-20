languages=(haskell lisp)
words=( awesome cool fun happy helpful interesting abstract category pure theory )

echo language > grep_irc_temp.txt
echo sum >> grep_irc_temp.txt
for i in "${words[@]}"
do
    echo $i >> grep_irc_temp.txt
done

echo newline >> grep_irc_temp.txt
for j in "${languages[@]}"
do
    echo $j >> grep_irc_temp.txt
    curl -s http://ircbrowse.net/browse/$j | grep 'class=\"description\"' | egrep -o '[0-9,]+ results' | sed 's/ results//' | sed 's/,//g' >> grep_irc_temp.txt
    for i in "${words[@]}"
    do
        echo $i $j
        curl -s http://ircbrowse.net/browse/$j?q=$i | grep 'class=\"description\"' | egrep -o '[0-9,]+ results' | sed 's/ results//' | sed 's/,//g' >> grep_irc_temp.txt
    done
    echo newline >> grep_irc_temp.txt
done

cat grep_irc_temp.txt | perl -pe "s/\n/,/g" | perl -pe "s/newline/\n/g" | perl -pe "s/^,(.*),/\1/g" | perl -pe "s/^,//g" | perl -pe "s/language/subreddit/g" > analysis/irc_words.csv
rm grep_irc_temp.txt