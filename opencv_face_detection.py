# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:31:24 2021

@author: ahmthaydrornk
"""

import cv2

YUZ_BULAN_MODEL = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

KAMERA = cv2.VideoCapture(1)

"""
Kamera'dan sürekli olarak görüntü okumak için sonsuz bir döngü oluşturuyoruz.
"""

while True:

    """ 
    .read() fonksiyonu ile KAMERA'dan anlık bir görüntü okunur ve
    bu görüntü OKUNAN_RENKLI_GORUNTU değişkenine aktarılır. Sorunsuz bir 
    şekilde okunursa OKUNDU_BILGISI True olarak dönecektir.
    """

    OKUNDU_BILGISI, OKUNAN_RENKLI_GORUNTU = KAMERA.read()

    """
    Haar Cascade modeli yalnızca gri seviyeli görüntülerde yüz tespiti 
    yapmaktadır. read() fonksiyonu ile okuduğumuz görüntü renkli bir görüntü
    olduğundan öncelikle cvtColor(, cv2.COLOR_BGR2GRAY) fonksiyonu ile 
    gri seviyeye çevrilmektedir.
    """
    GRI_GORUNTU = cv2.cvtColor(OKUNAN_RENKLI_GORUNTU, cv2.COLOR_BGR2GRAY)

    """
    detectMultiScale(GRI_GORUNTU, 1.1, 4) fonksiyonu ile gri görüntü
    içerisindeki yüz görüntüleri bulunmaktadır.
    """
    BULUNAN_YUZLER = YUZ_BULAN_MODEL.detectMultiScale(GRI_GORUNTU, 1.1, 4)

    """
    Bulunan her bir yüz görüntüsü 4 adet özelliğe sahiptir;
    x: bulunan yüzün sol üst köşesinin x koordinatı
    y: bulunan yüzün sol üst köşesinin y koordinatı
    w: bulunan yüzün yatay olarak genişliği
    h: bulunan yüzün dikey olarak uzunluğu
    """
    for (x, y, w, h) in BULUNAN_YUZLER:

        """
        Koordinatlar gri seviyeli görüntü üzerinde bulunduğu anda gri seviyeli
        görüntüyü tekrar kullanmayız. Onun yerine bulunan yüz etrafına bir
        dikdörtgen çizmek için okunan renkli görüntüyü kullanırız.
        cv2.rectangle() fonksiyonu verilen koordinatlara göre dikdörtgen
        oluşturacaktır.
        """
      
        cv2.rectangle(OKUNAN_RENKLI_GORUNTU, (x, y), (x+w, y+h), (255, 255, 0), 2)

    """ Görüntü yeni açılan pencerede gösterilir """
    cv2.imshow('YUZ BULMA UYGULAMASI', OKUNAN_RENKLI_GORUNTU)

    """ 
    Burada 27 ESC'ye karşılık gelmektedir. Programı durdurmak için ESC tuşuna
    basmalıyız aksi taktirde program kendini durduracaktır.
    """
    if cv2.waitKey(30) & 0xff == 27:
        
        cv2.destroyAllWindows()
        
        break

""" Tüm işlemler bittikten sonra kameranın kapanması için bu komut eklenir. """
KAMERA.release()
