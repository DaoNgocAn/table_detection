# -*- coding: utf-8 -*-

class Content:
    """Logical page content.

    @remark This an abstract base class.
    """
    def __init__(self, label, lid, plid, box):
        """Constructor of content.
        
        @param  label label of content
        @param  lid logical ID
        @param  plid parent's logical ID
        @param  box bounding box
        """
        self._label = label
        self._box = box
        self._lid = lid
        self._plid = plid

    def getLabel(self):
        return self._label
        
    def getBox(self):
        return self._box

    def setBox(self, box):
        self._box = box

    def getLid(self):
        return self._lid

    def getPlid(self):
        return self._plid

    def setPlid(self, plid):
        self._plid = plid

    def isLeaf(self):
        """Tell whether object is Leaf or not.
        @return True if object is leaf; False elsewise.
        """
        raise NotImplementedError

    def __str__(self):
        return '[Content, label: %s, lid: %d, plid: %s]' %     \
               (self._label, self._lid, str(self._plid))

class Leaf(Content):
    """A leaf content has a physical id and doesn't contain other contents."""
    def __init__(self, label, pid, lid, plid, box):
        Content.__init__(self, label, lid, plid, box)
        self.setPid(pid)

    def isLeaf(self):
        """Always return True."""
        return True

    def setPid(self, pid):
        self._pid = pid
        
    def getPid(self):
        """Return physical ID of object.
        @return objcet's physical ID
        @remark For subtle reasons, physical ID are strings.
        """
        return self._pid

    def __str__(self):
        return '[%s, pid: %s]' %        \
               (Content.__str__(self), self._pid)

class Composite(Content):
    """A composite content contains other contents."""
    def __init__(self, label, lid, plid, clids, box):
        Content.__init__(self, label, lid, plid, box)
        self.setClids(clids)

    def isLeaf(self):
        """Always return False."""
        return False

    def getClids(self):
        """Return lids of children.
        
        @return a list of children's logical ID's.
        
        @remark a copy is returned, modification on the returned list will not affect
        the object.
        """
        return self._clids[:]

    def setClids(self, clids):
        """Set lids of children.
        
        @param  clids new list of children's ID's
        
        @remark this method saves a copy of the parameter.
        """
        self._clids = clids[:]

    def __str__(self):
        return '%s, clids: %s' %        \
               (Content.__str__(self), str(self._clids))
