"""원본을 건드리지 않고 손상된 연결과 ZIP 데이터가 검출되는지 확인한다."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_network_structure import audit
import read_aihub_split_zip as reader


class ArchiveChecks(unittest.TestCase):
    def make_archive(self, directory):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("신호최적화데이터/부천시/net.xml", "<net>한글</net>")
        data = stream.getvalue()
        pos = data.rfind(b"PK\x05\x06")
        end = struct.unpack_from("<4s4H2IH", data, pos)
        p = directory / "central.bin"
        p.write_bytes(data[end[6]:end[6] + end[5]])
        return data, p, list(reader.parse_directory(p))

    def test_utf8_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, _, entries = self.make_archive(Path(tmp))
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                self.assertEqual(entries[0]["size"], z.infolist()[0].file_size)
                self.assertEqual(entries[0]["crc32"], z.infolist()[0].CRC)
                self.assertEqual(entries[0]["name"], z.namelist()[0])

    def test_truncated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, p, _ = self.make_archive(Path(tmp))
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(EOFError):
                list(reader.parse_directory(p))

    def test_extract_and_crc_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); data, _, entries = self.make_archive(d)
            (d / "bucheon_entries.json").write_text(json.dumps(entries))
            class MemoryVolumes:
                def __init__(self, _): pass
                def read(self, disk, offset, size): return data[offset:offset+size]
            with patch.object(reader, "DriveVolumes", MemoryVolumes):
                reader.extract(d, [entries[0]["name"]])
                target=d / "original_selected" / entries[0]["name"]
                self.assertEqual(target.read_text(), "<net>한글</net>")
                target.write_text("corrupted")
                with self.assertRaisesRegex(ValueError, "검증 실패"):
                    reader.extract(d, [entries[0]["name"]])

    def test_read_across_part_and_volume_boundaries(self):
        v=reader.DriveVolumes.__new__(reader.DriveVolumes)
        v.starts=[0,4,8,12,16,20]
        v.volumes=[(key,[{"offset":0,"size":2,"name":"a"},{"offset":2,"size":2,"name":"b"}],4) for key in reader.KEYS]
        blobs={f"{reader.BASE}/{key}/{name}":bytes(range(i*4+j*2,i*4+j*2+2)) for i,key in enumerate(reader.KEYS) for j,name in enumerate(["a","b"])}
        def read(cmd, **_):
            skip=int(cmd[cmd.index('--offset')+1]);size=int(cmd[cmd.index('--count')+1])
            return blobs[cmd[2]][skip:skip+size]
        with patch.object(reader.subprocess,"check_output",read):
            self.assertEqual(v.read(0,1,15),bytes(range(1,16)))
            with self.assertRaises(ValueError):v.read(4,0,5)


class NetworkChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.source=Path(self.temp.name)/'valid.net.xml'
        self.source.write_text('''<net>
          <edge id="477610475_01" from="entry" to="junction"><lane id="a_0" index="0" speed="10" length="100"/></edge>
          <edge id="b" from="junction" to="exit"><lane id="b_0" index="0" speed="10" length="100"/></edge>
          <edge id=":j_0" function="internal"><lane id=":j_0_0" index="0" speed="10" length="10"/></edge>
          <junction id="entry" type="dead_end"/>
          <junction id="junction" type="traffic_light" incLanes="a_0" intLanes=":j_0_0"/>
          <junction id="exit" type="dead_end" incLanes="b_0"/>
          <connection from="477610475_01" to="b" fromLane="0" toLane="0" via=":j_0_0" tl="signal" linkIndex="0"/>
          <connection from=":j_0" to="b" fromLane="0" toLane="0"/>
          <tlLogic id="signal"><phase duration="10" state="G"/><phase duration="3" state="y"/></tlLogic>
        </net>''')

    def tearDown(self):
        self.temp.cleanup()

    def changed_report(self, modify):
        with tempfile.TemporaryDirectory() as tmp:
            tree=ET.parse(self.source);modify(tree.getroot())
            path=Path(tmp)/'fixture.net.xml';tree.write(path)
            return audit(path)

    def test_known_model_valid(self):
        r=audit(self.source)
        self.assertEqual(r['structural_errors'],[])
        self.assertEqual(r['internal_sink_edges'],[])

    def test_unknown_via(self):
        r=self.changed_report(lambda n:n.find('connection').set('via','missing'))
        self.assertIn('invalid_connection_reference',{x['type'] for x in r['structural_errors']})

    def internal_duplicate_index_report(self, invalid_ordinal=False):
        def modify(n):
            a=n.find('edge'); ET.SubElement(a,'lane',id='a_1',index='1',speed='10',length='100')
            b=next(e for e in n.findall('edge') if e.get('id')=='b')
            ET.SubElement(b,'lane',id='b_1',index='1',speed='10',length='100')
            internal=next(e for e in n.findall('edge') if e.get('id')==':j_0')
            ET.SubElement(internal,'lane',id=':j_0_1',index='0',speed='10',length='10')
            ET.SubElement(n,'connection',{'from':'477610475_01','to':'b','fromLane':'1','toLane':'1','via':':j_0_1'})
            ET.SubElement(n,'connection',{'from':':j_0','to':'b','fromLane':'2' if invalid_ordinal else '1','toLane':'1'})
            for j in n.findall('junction'):
                if j.get('id')=='junction':j.set('incLanes','a_0 a_1');j.set('intLanes',':j_0_0 :j_0_1')
                if j.get('id')=='exit':j.set('incLanes','b_0 b_1')
        return self.changed_report(modify)

    def test_duplicate_internal_lane_index_metadata_is_review_only(self):
        r=self.internal_duplicate_index_report()
        self.assertEqual(r['structural_errors'],[])
        mismatch=[x for x in r['review_findings'] if x['type']=='lane_index_metadata_mismatch']
        self.assertEqual(len(mismatch),1)
        self.assertEqual(mismatch[0]['declared_index'],'0')
        self.assertEqual(mismatch[0]['runtime_ordinal'],1)

    def test_true_out_of_range_internal_lane_still_fails(self):
        r=self.internal_duplicate_index_report(invalid_ordinal=True)
        self.assertTrue(any(x['type']=='invalid_connection_reference' and x.get('field')=='from_lane'
                            for x in r['structural_errors']))

    def test_bad_signal_index(self):
        def modify(n):next(c for c in n.findall('connection') if c.get('tl')).set('linkIndex','9999')
        r=self.changed_report(modify)
        self.assertIn('signal_index_out_of_range',{x['type'] for x in r['structural_errors']})

    def test_unknown_junction_lane(self):
        r=self.changed_report(lambda n:n.find('junction').set('incLanes','missing'))
        self.assertIn('unknown_junction_lane',{x['type'] for x in r['structural_errors']})

    def test_removed_successors_detected(self):
        edge='477610475_01'
        def modify(n):
            for c in list(n.findall('connection')):
                if c.get('from')==edge:n.remove(c)
        r=self.changed_report(modify)
        self.assertIn(edge,r['internal_sink_edges'])
        self.assertIn(edge,r['vehicle_class_reachability']['passenger']['cannot_reach_boundary_exit'])


if __name__=='__main__':unittest.main()
