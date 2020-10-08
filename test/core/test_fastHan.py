import unittest
from fastHan import FastHan


class TestFastHan(unittest.TestCase):  # 继承了unittest.TestCase这样可以直接利用unittest的一些function
    def test_init(self):
        # 测试是否可以正确initialize

        model = FastHan()

        model = FastHan('large')

    def test_call(self):
        
        sentence=['一行人下得山来，走不多时，忽听前面猛兽大吼之声一阵阵的传来。',
        '韩宝驹一提缰，胯下黄马向前窜出，奔了一阵，忽地立定，不论如何催迫，黄马只是不动。',
        '韩宝驹心知有异，远远望去，只见前面围了一群人，有几头猎豹在地上乱抓乱扒。'
        '他知坐骑害怕豹子，跃下马来，抽出金龙鞭握在手中。']

        targets=['CWS','POS','NER','Parsing']
        model = FastHan('large')
        for target in targets:
            model(sentence,target)
