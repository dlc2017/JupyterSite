
# Neural Machine Translation for IPO Documents
### LUO Linkai
### *Deep Learning Research & Application Centre*
### August 2017

<script>
    var code_show=true; //true -> hide code at first

    function code_toggle() {
        $('div.prompt').hide(); // always hide prompt

        if (code_show){
            $('div.input').hide();
        } else {
            $('div.input').show();
        }
        code_show = !code_show
    }
    $( document ).ready(code_toggle);
</script>


```python
def load_data(file, ids):
    with open(file, 'r') as f:
        data = f.read().split('\n')
    if len(ids) == 2 and ids[0] < ids[1]:
        return data[ids[0]:ids[1]]
    else:
        #return [data.split('\n')[i] for i in ids]
        return [e for i, e in enumerate(data) if i in ids]
width = 80
```

# Our Goal
* <font color='blue'>Deep-learning-initiated</font> neural machine translation (NMT) system for IPO documents

* Fully automatic high-quality MT is still <font color='red'>a distant goal </font>
* Good <font color='red'> assistant </font> to human translator
* <font color='red'> Better </font> (compoarable) quality than Google translate
* Full <font color='red'> control </font> of files

# Google Translate?


```python
g_ids = [11, 502, 200]
source1 = load_data('./data/test.src', g_ids)
reference1 = load_data('./data/test.trg', g_ids)
target1 = ['此外，我們不能向你保證，其他人不會通過獨立發展或其他法律手段獲得這些商業秘密的知識。',
'截至最近切實可行日期，日本法定股本為億萬股，分為200股。']

#for src in source1:
#       print('{}\n'.format(src.replace('@@ ', '')))
#print('-'*100)
source = ['松下問童子','our company has no outstanding convertible debt securities as of the latest practicable date.']+source1
target = ['Matsushita asked the boy', '我們公司沒有尚未轉換債券作為最後實際可行日期。']+target1
reference = ['Beneath the pine, I asked of the child.', '截至最後實際可行日期，本公司並無任何尚未行使的可換股債務證券。']+reference1
source = source[1:]
target = target[1:]
reference = reference[1:]
for src, tgt, ref in zip(source, target, reference):
    print('Source   : {}\nGoogle   : {}\nReference: {}\n'.format(src.replace('@@ ', ''), tgt, ref.replace('@@ ', '').replace(' ', '')))
    print('-'*width)
```

    Source   : our company has no outstanding convertible debt securities as of the latest practicable date.
    Google   : 我們公司沒有尚未轉換債券作為最後實際可行日期。
    Reference: 截至最後實際可行日期，本公司並無任何尚未行使的可換股債務證券。
    
    ------------------------------------------------------------
    Source   : our company has no outstanding convertible debt securities as of the latest practicable date .
    Google   : 此外，我們不能向你保證，其他人不會通過獨立發展或其他法律手段獲得這些商業秘密的知識。
    Reference: 截至最後實際可行日期，本公司並無任何尚未行使的可換股債務證券。
    
    ------------------------------------------------------------
    Source   : in addition , we can not assure you that others will not obtain knowledge of these trade secrets through independent development or other legal means .
    Google   : 截至最近切實可行日期，日本法定股本為億萬股，分為200股。
    Reference: 此外，我們無法向閣下保證他人不會透過自主開發或以其他合法途徑獲悉該等商業機密。
    
    ------------------------------------------------------------


# Neural Machine Translation System
* **Data**: ~2 million paired sentences
    * train/dev/test: 80%/10%/10%
* **Parameters**: ~1.2 G
* **Equipment**: CentOS, 1T RAM, Intel(R) Xeon(R) 2.10GHz, NVIDIA Tesla P100 GPU --> **6.0x** faster !
* **Training time**: ~14 days (Two ideas tested)
* **Decode (translate) time**: ~8s/80 sentences (without GPU), ~2s/80 sentences (with GPU)
* *Applied Deep Learning is an iterative process !*

<div id="image-table">
    <table cellspacing="10">
        <tr>
            <td style="padding:5px">
                <img width="400", src="./img/nmt_iterative.png">
              </td>
            <td style="padding:5px">
                <img width="400", src="./img/BLEU_2017_8_18.png">
             </td>
        </tr>
    </table>
</div>
<p> **Nerural Machine Translation Development is an iterative process** </p>

# Some translation examples


```python
g_ids = [11, 502, 200, 102, 208, 13, 800]
source = load_data('./data/test.src', g_ids)
target = load_data('./data/large_test_120000.de', g_ids)
reference = load_data('./data/test.trg', g_ids)
google = ['本公司於最近實際可行日期無債務證券。',
'此外，還發布了2009年8月加強施工用地管理和促進利用未經批准使用的通知，重申了現行的閒置用地規定。',
'該通知必須指明會議的時間和地點，在特殊業務的情況下，該業務的一般性質。',
'此外，我們不能向你保證，其他人不會通過獨立發展或其他法律手段獲得這些商業秘密的知識。',
'行使其根據公司章程賦予的其他權利。',
'截至最近切實可行日期，日本法定股本為億萬股，分為200股。',
'關於是否行使購買新業務機會的選擇權的決定將由獨立非執行董事作出，以確保該決定適當考慮到獨立股東的利益。']

len_per_slide = 4
source1 = source[:len_per_slide]
target1 = target[:len_per_slide]
reference1 = reference[:len_per_slide]
google1 = google[:len_per_slide]

#print('-'*100)
for src, tgt, ref, gl in zip(source1, target1, reference1, google1):
    print('Source   : {}\nTranslate: {}\nReference: {}\nGoogle   : {}'.format(src.replace('@@ ', ''), tgt.replace('@@ ', '').replace(' ', ''), ref.replace('@@ ', '').replace(' ', ''), gl))
    print('-'*width)
```

    Source   : our company has no outstanding convertible debt securities as of the latest practicable date .
    Translate: 截至最後實際可行日期，本公司並無已發行的可換股債務證券。
    Reference: 截至最後實際可行日期，本公司並無任何尚未行使的可換股債務證券。
    Google   : 本公司於最近實際可行日期無債務證券。
    --------------------------------------------------------------------------------
    Source   : furthermore , mlr issued the notice on strengthening administration of construction land and promoting the utilisation of approved land without utilisation in august 2009 , which reiterates the current rules regarding idle land .
    Translate: 此外，國土資源部於二零零九年八月發佈《關於加強建設用地管理有關問題的通知》，通知重申閒置土地的現行規則。
    Reference: 此外，國土資源部於2009年8月發出《關於嚴格建設用地管理促進批而未用土地利用的通知》，重申了對閒置土地的現行規則。
    Google   : 此外，還發布了2009年8月加強施工用地管理和促進利用未經批准使用的通知，重申了現行的閒置用地規定。
    --------------------------------------------------------------------------------
    Source   : the notice must specify the time and place of the meeting and , in the case of special business , the general nature of that business .
    Translate: 通告須註明舉行會議的時間及地點，倘有特別事項，則須註明有關事項的一般性質。
    Reference: 通告須註明舉行會議之時間及地點，倘有特別事項，則須註明有關事項之一般性質。
    Google   : 該通知必須指明會議的時間和地點，在特殊業務的情況下，該業務的一般性質。
    --------------------------------------------------------------------------------
    Source   : in addition , we can not assure you that others will not obtain knowledge of these trade secrets through independent development or other legal means .
    Translate: 此外，我們無法向閣下保證其他人士不會透過獨立發展或其他法律方式取得該等商業機密的認識。
    Reference: 此外，我們無法向閣下保證他人不會透過自主開發或以其他合法途徑獲悉該等商業機密。
    Google   : 此外，我們不能向你保證，其他人不會通過獨立發展或其他法律手段獲得這些商業秘密的知識。
    --------------------------------------------------------------------------------



```python
source2 = source[len_per_slide:]
target1 = target[len_per_slide:]
reference1 = reference[len_per_slide:]
google1 = google[len_per_slide:]

#print('-'*100)
for src, tgt, ref, gl in zip(source1, target1, reference1, google1):
    print('Source   : {}\nTranslate: {}\nReference: {}\nGoogle   : {}'.format(src.replace('@@ ', ''), tgt.replace('@@ ', '').replace(' ', ''), ref.replace('@@ ', '').replace(' ', ''), gl))
    print('-'*width)
```

    Source   : our company has no outstanding convertible debt securities as of the latest practicable date .
    Translate: 及行使組織章程細則賦予彼等的其他權利。
    Reference: 及行使組織章程細則賦予彼等的其他權利。
    Google   : 行使其根據公司章程賦予的其他權利。
    --------------------------------------------------------------------------------
    Source   : furthermore , mlr issued the notice on strengthening administration of construction land and promoting the utilisation of approved land without utilisation in august 2009 , which reiterates the current rules regarding idle land .
    Translate: 於最後實際可行日期，HHGraceJapan的法定股本為10,000,000日圓，分為200股股份。
    Reference: 於最後實際可行日期，HHGraceJapan的法定股本為10,000,000日圓，分為200股股份。
    Google   : 截至最近切實可行日期，日本法定股本為億萬股，分為200股。
    --------------------------------------------------------------------------------
    Source   : the notice must specify the time and place of the meeting and , in the case of special business , the general nature of that business .
    Translate: 是否行使選擇權或不能收購新業務機會選擇權將由獨立非執行董事作出，以確保就獨立股東的利益作出適當考慮。
    Reference: 是否行使接納新業務機會選擇權的決定將由獨立非執行董事作出，以確保該決定將充分考慮我們獨立股東的利益。
    Google   : 關於是否行使購買新業務機會的選擇權的決定將由獨立非執行董事作出，以確保該決定適當考慮到獨立股東的利益。
    --------------------------------------------------------------------------------


# Some existing problems
* <font color='blue'> Polluted data </font>
    * unwanted symbols
    * unpaired sentences
* <font color='blue'> Longer sentences </font>
    * insufficient translation


```python
b_ids = [209, 509, 608]
source = load_data('./data/test.src', b_ids)
target = load_data('./data/large_test_120000.de', b_ids)
reference = load_data('./data/test.trg', b_ids)
google = ['根據重組，2011年3月30日，Teel將其份額轉讓給teebvil，代價為1港幣。',
'在中華人民共和國的時候 常務委員會宣布，加入紐約公約。',
'我們的董事在考慮到新業務機會的盈利能力，風險和業務策略後，會否考慮到這樣的機會是否符合我們集團整體的最佳利益，決定是否行使先發製人的權利。']

#print('-'*100)
for src, tgt, ref, gl in zip(source, target, reference, google):
    print('Source   : {}\nTranslate: {}\nReference: {}\nGoogle   : {}'.format(src.replace('@@ ', ''), tgt.replace('@@ ', '').replace(' ', ''), ref.replace('@@ ', '').replace(' ', ''), gl))
    print('-'*width)

```

    Source   : pursuant to the reorganisation , on 30 march 2011 teel transferred its share in peil to teebvil for a consideration of hk $ 1 .
    Translate: 根據重組，二零一一年三月三十日，TEEL以代價1。
    Reference: 根據重組，於二零一一年三月三十日，TEEL以代價1。
    Google   : 根據重組，2011年3月30日，Teel將其份額轉讓給teebvil，代價為1港幣。
    ------------------------------------------------------------
    Source   : at the time of the prc &apos; s accession to the new york convention , the standing committee declared that .
    Translate: 在中國加入《紐約公約》的同時，全國人大常委會宣佈。
    Reference: 常務委員會於中國加入紐約公約時同時宣稱。
    Google   : 在中華人民共和國的時候 常務委員會宣布，加入紐約公約。
    ------------------------------------------------------------
    Source   : our directors will , after taking into consideration the profitability , risks and business strategies of the new business opportunity , and whether such opportunity is in the best interest of our group as a whole , determine if we shall exercise such pre-emptive rights .
    Translate: 經考慮我們是否行使該優先受讓權。
    Reference: 董事將根據新商業機會的盈利能力、風險及業務戰略以及該機會是否符合本集團的整體最佳利益後，審核決定是否行使該優先受讓權。
    Google   : 我們的董事在考慮到新業務機會的盈利能力，風險和業務策略後，會否考慮到這樣的機會是否符合我們集團整體的最佳利益，決定是否行使先發製人的權利。
    ------------------------------------------------------------



```python
for src in source:
    print('{}'.format(src.replace('@@ ', '').replace('\n', '')))
```

    pursuant to the reorganisation , on 30 march 2011 teel transferred its share in peil to teebvil for a consideration of hk $ 1 .
    at the time of the prc &apos; s accession to the new york convention , the standing committee declared that .
    our directors will , after taking into consideration the profitability , risks and business strategies of the new business opportunity , and whether such opportunity is in the best interest of our group as a whole , determine if we shall exercise such pre-emptive rights .


# Our plan
* Data collection & processing
    * More data: Annual report from HKEX; Non-IPO documents
    * Data cleaning, etc
    * Difficulty in processing Alpha data. Advices?
* Model improvement
    * New ideas
    * Training tricks
* Interface
    * Features

# More examples


```python

begin = 10
num_sent = 5
ids = [begin, begin+num_sent]
source = load_data('./data/test.src', ids)
target = load_data('./data/test.trg', ids)
decode = load_data('./data/large_test_120000.de', ids)

#for i in range(num_sent-10):
#    print(source[i].replace('@@', ''))


for i in range(len(source)):
    print('Source {}: {}'.format(begin+i, source[i].replace('@@ ', '')))
    print('Translate: {}'.format(decode[i].replace('@@ ', '').replace(' ', '')))
    print('Reference: {}'.format(target[i].replace('@@ ', '').replace(' ', '')))
    print('-'*width)
```

    Source 10: under the arrangements currently in force , the aggregate emoluments payable by our group to and benefits in kind receivable by our directors for the year ending 31 december 2013 are expected to be approximately rmb3,204,000 .
    Translate: 根據現行安排，截至二零一三年十二月三十一日止年度，本集團應付酬金及董事應收實物利益總額預期約為人民幣3,204,000元。
    Reference: 根據現行安排，截至2013年12月31日止年度，本集團應付董事的薪酬總額及上述董事應收的實物利益預期約為人民幣3,204,000元。
    ------------------------------------------------------------
    Source 11: our company has no outstanding convertible debt securities as of the latest practicable date .
    Translate: 截至最後實際可行日期，本公司並無已發行的可換股債務證券。
    Reference: 截至最後實際可行日期，本公司並無任何尚未行使的可換股債務證券。
    ------------------------------------------------------------
    Source 12: the difference between the accumulated global installed wind power capacity for any two consecutive years is not equal to the newly installed global wind power capacity for the more recent of the same two years because some of the already installed wtgs were decommissioned .
    Translate: 因部分已安裝裝機容量停牌，累計風電裝機容量與最近兩年的累計風電裝機容量與其他兩個年度有所不同。
    Reference: 任何連續兩年世界累計風電裝機容量之間的差額並不等於該兩年的較近期的世界新增風電裝機容量，原因是部分已安裝的風力發電機組退役停用。
    ------------------------------------------------------------
    Source 13: furthermore , mlr issued the notice on strengthening administration of construction land and promoting the utilisation of approved land without utilisation in august 2009 , which reiterates the current rules regarding idle land .
    Translate: 此外，國土資源部於二零零九年八月發佈《關於加強建設用地管理有關問題的通知》，通知重申閒置土地的現行規則。
    Reference: 此外，國土資源部於2009年8月發出《關於嚴格建設用地管理促進批而未用土地利用的通知》，重申了對閒置土地的現行規則。
    ------------------------------------------------------------
    Source 14: as at the latest practicable date , the group has obtained all material intellectual property rights for its operations and is the registered proprietor and beneficial owner of the following material trademarks .
    Translate: 於最後實際可行日期，本集團已就其經營取得所有重要的知識產權，並為以下重要商標的註冊所有人及實益擁有人。
    Reference: 於最後可行日期，本集團已取得其營運所需的一切重大知識產權，並為下列重大商標的註冊擁有人及實益擁有人。
    ------------------------------------------------------------

