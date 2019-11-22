# -*- coding: utf-8 -*-

"""This file defines the Page class and possible Errors might be thrown."""

from dataset.Content import Content, Leaf, Composite
from dataset.Rect import Rect
from dataset.Labels import *

class PageError(Exception): pass
class LabelError(PageError): pass
class DupLidError(PageError): pass
class DupPidError(PageError): pass
class AttachBeforeDetachError(PageError): pass
class DeleteBeforeDetachError(PageError): pass
class DeleteNotExistError(PageError): pass

class Page:
    """Logical page. This class is a container of contents.
    
    @remarks Contents are created _inside_ the Page object,
    and should be deleted _inside_ the Page object.\n
    An instance of Page class stores the
        
        lidIndex:       {lid: content}          \n
        pidIndex:       {pid: content}          \n
        labelIndex:     {label: [lid]}          \n
        
    relations.
    """
    def __init__(self):
        self._lidIndex = {}
        self._labelIndex = {}
        self._pidIndex = {}
        self._maxLid = 0

    def setPageNum(self, pgno):
        """Set page number attribute of current page."""
        self._pgno = pgno

    def getPageNum(self):
        """See getPageNum."""
        return self._pgno

    def setPageType(self, pgtype):
        """Set page type attribute of current page."""
        self._pgtype = pgtype

    def getPageType(self):
        """See setPageType."""
        return self._pgtype

    def setCropBox(self, cropbox):
        """Set display area of current page.
        @remark The origin is the left-bottom corner of display area.
        """
        self._cropbox = cropbox

    def getCropBox(self):
        """See setCropBox."""
        return self._cropbox

    def getByLid(self, lid):
        """Return content with a certain lid.
        
        @param  lid logical ID of object
        @return content object with lid as logical ID;
        If such object is not found, None is returned.
        """
        return self._lidIndex.get(lid, None)

    def getByPid(self, pid):
        """Return leaf content with a certain pid. Only for Leaf objects.
        
        @param  pid physical ID of object
        @return Leaf content object with pid as physical ID.\n
        If such object is not found, None is returned.
        """
        return self._pidIndex.get(pid, None)        

    def getByLabel(self, label):
        """Return contents with a certain label.
        
        @param  label label of the desired contents.
        @return a list of content objects.
        """
        lids = self._labelIndex.get(label)
        if not lids:
            return []

        contents = []
        for lid in lids:
            content = self.getByLid(lid)
            if content.isLeaf() or content.getClids():
                contents.append(content)
        return contents

    def count(self):
        """
        @return Number of content objects in the page.
        """
        return len(self._lidIndex)
    

    def create(self, label, pid=None, lid=None, plid=None, clids=None, box=None):
        """Create a new content object.
        
        @param  label Label of new object
        @param  pid Physical ID of new object
        @param  lid Logical ID of new object
        @param  plid Parent's logical ID
        @param  clids A list of clids children's logical ID's
        @param  box bounding box of new object
        @return A new Leaf or Composite object.
        
        @remark Whether the returned object is Leaf or Compsite is decided by the
        label parameter.\n
        If label doesn't appear in tupples LABEL_LEAF or LABEL_COMPOSITE in
        Labels.py, LabelError is raised.\n
        If an object with same pid is already in the page, DupPidError is raised.\n
        If an object with same lid is already in the page, DupLidError is raised.\n
        If the lid parameter is 0 or None, a proper ID is generated.
        This should never lead to DupLidError.\n

        If the object is created successfully, it can be indexed through its logical
        ID.
        """
        if not lid:
            self._maxLid += 1
            lid = self._maxLid
        content = None
        if label in LABEL_LEAF:
            content = self._createLeaf(label, pid, lid, plid, box)
        elif label in LABEL_COMPOSITE:
            content = self._createComposite(label, lid, plid, clids, box)
        else:
            raise LabelError("Label '%s' is not acceptable." % label)
        if content:
            self._add(content)
        return content

    def _createLeaf(self, label, pid, lid, plid, box):
        return Leaf(label, pid, lid, plid, box)

    def _createComposite(self, label, lid, plid, clids, box):
        return Composite(label, lid, plid, clids, box)

    def delete(self, lid):
        """Wipe the object with logical ID lid out of this page.
        
        @param  lid Logical ID of target object
        
        @remark If no object can be indexed by lid, DeleteNotExistError is raised.\n
        If the object being deleted is not detached from its parent or children,
        DeleteBeforeDetachError is raised.
        """
        self._remove(lid)

    def _add(self, content):
        lid = content.getLid()
        if lid in self._lidIndex:
            raise DupLidError
        self._lidIndex[lid] = content

        label = content.getLabel()
        if label not in self._labelIndex:
            self._labelIndex[label] = set()
        self._labelIndex[label].add(lid)
        self._maxLid = max(self._maxLid, lid)

        if content.isLeaf():
            pid = content.getPid()
            if pid in self._pidIndex:
                raise DupPidError
            self._pidIndex[pid] = content

    def _remove(self, lid):
        content = self.getByLid(lid)
        if not content:
            raise DeleteNotExistError(
                  'content %d is not contained in this page' % lid)
        if content.getPlid() or (not content.isLeaf() and content.getClids()):
            raise DeleteBeforeDetachError(
                  'content %d should be detached before deletion.' % lid)
        # remove content from page
        label = content.getLabel()
        self._labelIndex[label].remove(lid)
        self._lidIndex.pop(lid)

    def attach(self, plid, clids):
        """Attach children to parent.
        
        @param  plid parent's ID
        @param  clids children's ID's
        
        @remark If A, B, C, D are all ID's, and B is A's child,\n
        after self.attach(A, [C, D]), A is parent of B, C and D.\n
        If a child is attached to a different parent, AttachBeforeDetachError
        is raised.
        """
        
        if not plid or not clids:
            return
        
        # check if parent is not leaf
        parent = self._lidIndex[plid]
        assert not parent.isLeaf()

        # update parent id of children contents
        for clid in clids:
            child = self.getByLid(clid)
            # child has no parent or its parent is plid
            if child and child.getPlid() and child.getPlid() != plid:
                raise AttachBeforeDetachError(                                  \
                      'child %d already has a parent %d diffrent from %d ' %    \
                      (clid, child.getPlid(), plid))
            if child:
                child.setPlid(plid)

        # update clids of parent content
        oclids = parent.getClids()
        oset = set(oclids)
        for clid in clids:
            if not clid in oset:
                oclids.append(clid)
        parent.setClids(oclids)
        self.update(plid)

    def detach(self, plid, clids):
        """Detach children from parent.
        
        @param  plid parent's ID.
        @param  clids children's ID's
        
        @remark After detaching, children's plid become 0, and clids are removed
        from parent's list of children's ID's.
        """
        parent = self._lidIndex[plid]
        assert not parent.isLeaf()

        # update parent id of children contents
        for clid in clids:
            child = self._lidIndex[clid]
            if child.getPlid() == plid:
                child.setPlid(0)

        # update clids of parent content
        oclids = parent.getClids()
        for clid in clids:
            oclids.remove(clid)
        parent.setClids(oclids)
        self.update(plid)

    def update(self, lid):
        """Update a content node.
        
        @param  lid logical ID of the node/object
        
        @remark For now, only bounding box is updated.\n
        The update action propagates to its ancestors.
        """
        if lid not in self._lidIndex:
            return
        # assert lid in self._lidIndex
        content = self.getByLid(lid)
        clids = content.getClids()
        plid = content.getPlid()        

        x0 = y0 = float('inf')
        x1 = y1 = float('-inf')
        
        for clid in clids:
            child = self.getByLid(clid)
            if child:
                cbox = child.getBox()
                x0 = min(x0, cbox.x0())
                y0 = min(y0, cbox.y0())
                x1 = max(x1, cbox.x1())
                y1 = max(y1, cbox.y1())
        content.setBox(Rect(x0, y0, x1, y1))
        
        self.update(plid)

    def lidToPids(self, lid):
        """Maps logical ID of an object to physical ID's contained in it.
        
        @param  lid logical ID of an object
        @return a list of physical ID's contained in corresponding object,
        no order garanteed.
        """
        def __pidGenerator(page, lid):
            content = page.getByLid(lid)
            if content.isLeaf():
                yield content.getPid()
            else:
                clids = content.getClids()
                for clid in clids:
                    for pid in page.lidToPids(clid):
                        yield pid
        return [pid for pid in __pidGenerator(self, lid)]

    def labelToPids(self, label):
        """Maps label to pids, no order garanteed.
        
        @param  label label of desired objects
        @return a list of physical ID's contained in corresponing objecs with
        this label, no order garanteed.
        """
        lids = [content.getLid() for content in self.getByLabel(label)]
        pids = []
        for lid in lids:
            pids.extend(self.lidToPids(lid))
        return pids

    def lidToAncestors(self, lid):
        """Given a logical id, return a content list of its ancestors.
        
        @param  lid logical ID of an object
        @return a list of contents consisting of the object's ancestors.
        """
        def __ancestorGenerator(page, content):
            while content.getPlid():
                content = page.getByLid(content.getPlid())
                yield content
        content = self.getByLid(lid)
        return [ancestor for ancestor in __ancestorGenerator(self, content)]
