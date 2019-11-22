# -*- coding: utf-8 -*-

from xml.sax.handler import ContentHandler
from xml.sax import parse
from struct import unpack

from dataset.Rect import Rect
from dataset.XMLNames import *
from dataset.Page import Page
from dataset.Labels import *

class PageReader:
    """Class that reads an xml file and generate a logical page tree."""

    @staticmethod
    def read(filepath):
        """Read an xml and create a page object.
        
        @param  filepath path to xml file
        @return logical page described by the xml file
        """
        page = Page()
        handler = PageHandler(page)
        parse(filepath, handler)
        return page

class PageHandler(ContentHandler):
    """Class that parses xml elements."""
    def __init__(self, page):
        self._page = page

    def startElement(self, name, attrs):
        """
        @param  name tag name of element
        @param  attrs attributes of the element
        """
        behaviors = {
            ELEM_PAGE: lambda: self._readPage(attrs),
            ELEM_LEAF: lambda: self._readContent(attrs),
            ELEM_COMPOSITE: lambda: self._readContent(attrs)
            }
        if name in behaviors:
            behaviors[name]()

    def endElement(self, name):
        """
        @param name tag name of element
        """
        pass

    def _decodeBox(self, string):
        def _hex2double(s):
            return unpack('>d', bytes.fromhex(s))[0]
        hexs = string.split(' ')
        return Rect(_hex2double(hexs[0]),
                    _hex2double(hexs[3]),
                    _hex2double(hexs[2]),
                    _hex2double(hexs[1]))

    def _readPage(self, attrs):
        pgno = attrs.get(ATTR_PAGENUM)
        self._page.setPageNum(pgno)

        pgtype = attrs.get(ATTR_PAGETYPE)
        self._page.setPageType(pgtype)

        cropbox = self._decodeBox(attrs.get(ATTR_PAGEBOX))
        self._page.setCropBox(cropbox)

    def _readContent(self, attrs):
        label = attrs.get(ATTR_LABEL)
        
        lid = int(attrs.get(ATTR_LID))
        # plid = int(attrs.get(ATTR_PLID)) if attrs.get(ATTR_PLID) else 0
        plid = 0
        box = self._decodeBox(attrs.get(ATTR_BBOX))
        if label in LABEL_LEAF:
            pid = self._readLeaf(attrs)
            self._page.create(label, pid=pid, lid=lid, plid=plid, box=box)
        else:
            clids = self._readComposite(attrs)
            self._page.create(label, lid=lid, plid=plid, clids=clids, box=box)
            self._page.attach(lid, clids)

    def _readLeaf(self, attrs):
        pid = attrs.get(ATTR_PID)
        return pid

    def _readComposite(self, attrs):
        clidsData = attrs.get(ATTR_CLIDS)
        clids = []
        if clidsData:
            clids = [int(clid) for clid in clidsData.split(' ')]
        return clids


if __name__ == '__main__':
    pr = PageReader()
    page = pr.read('/home/andn/PycharmProjects/table_detection/data/data_marmot/English/Positive/Labeled/10.1.1.1.2006_3.xml')
    import pprint
    pprint.pprint(page._pidIndex)

    print(page.getByLabel(LABEL_TABLEBODY))