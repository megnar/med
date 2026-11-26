"""Read-only biomedical tools for the extended MSK agent.

CIViC searches molecular-profile names, Reactome and SIGNOR accept UniProt IDs.
Results are capped and include source identifiers; no claim is made that returned
pathways or interactions establish causality for a patient's variant.
"""
import csv
import io
import json
import re
import threading
import time
from urllib.parse import quote

import requests

_UNIPROT = re.compile(r"^[A-Z0-9]{6,10}(?:-\d+)?$")
_CIVIC_LOCK = threading.Lock()
_CIVIC_LAST = 0.0

class BiomedicalTools:
    def __init__(self, timeout=25, max_results=5, session=None):
        self.timeout = timeout
        self.max_results = max_results
        self.session = session or requests.Session()

    def civic(self, profile):
        global _CIVIC_LAST
        profile = profile.strip()
        if not profile or len(profile) > 120:
            raise ValueError('Provide a molecular profile such as KRAS G12V (max 120 characters).')
        query = '''query($name:String!,$n:Int!){molecularProfiles(name:$name,first:$n){nodes{id name description link evidenceItems(first:3){nodes{id status evidenceType evidenceLevel evidenceDirection description disease{name} therapies{name} source{name citation publicationYear}}}}}}'''
        with _CIVIC_LOCK:
            time.sleep(max(0, 0.4 - (time.monotonic() - _CIVIC_LAST)))
            try:
                response = self.session.post('https://civicdb.org/api/graphql', json={'query':query,'variables':{'name':profile,'n':self.max_results}}, timeout=self.timeout)
            finally:
                _CIVIC_LAST = time.monotonic()
        response.raise_for_status()
        data = response.json()
        if data.get('errors'):
            raise RuntimeError('CIViC GraphQL error: ' + json.dumps(data['errors']))
        return {'source':'CIViC','query':profile,'matching':'name search; verify gene and variant; not necessarily exact', 'results':data['data']['molecularProfiles']['nodes']}

    def reactome(self, accession):
        accession = self._accession(accession)
        url='https://reactome.org/ContentService/data/mapping/UniProt/'+quote(accession,safe='')+'/pathways'
        response=self.session.get(url,timeout=self.timeout);response.raise_for_status()
        pathways=[{'id':p.get('stId'),'name':p.get('displayName'),'species':p.get('speciesName'),'inferred':p.get('isInferred'),'disease':p.get('isInDisease')} for p in response.json() if p.get('speciesName') in (None,'Homo sapiens')]
        return {'source':'Reactome','query':accession,'total':len(pathways),'results':pathways[:self.max_results]}

    def signor(self, accession):
        accession=self._accession(accession)
        response=self.session.get('https://signor.uniroma2.it/getData.php',params={'id':accession,'organism':'9606'},timeout=self.timeout);response.raise_for_status()
        rows=list(csv.reader(io.StringIO(response.text),delimiter='\t'))
        if rows and rows[0] and rows[0][0].upper()=='ENTITYA': rows=rows[1:]
        fields=('entity_a','type_a','id_a','database_a','entity_b','type_b','id_b','database_b','effect','mechanism','residue','sequence','tax_id','cell_data','tissue_data','modulator_complex','target_complex','modification_a','modification_a_sequence','modification_b','modification_b_sequence','pmid','direct','notes','annotator','sentence','signor_id','score')
        data=[dict(zip(fields,row)) for row in rows if len(row)>=10]
        return {'source':'SIGNOR','query':accession,'organism':'Homo sapiens','total':len(data),'results':data[:self.max_results]}

    @staticmethod
    def _accession(accession):
        accession=accession.strip().upper()
        if not _UNIPROT.fullmatch(accession):
            raise ValueError('Expected one UniProt accession, for example P04637 (TP53).')
        return accession

    def query(self, tool, argument):
        tool=tool.upper()
        if tool not in {'CIVIC','REACTOME','SIGNOR'}: raise ValueError('Unknown biomedical tool')
        start=time.monotonic()
        try:
            result=getattr(self,tool.lower())(argument)
            return {'ok':True,'tool':tool,'elapsed_seconds':round(time.monotonic()-start,3),'data':result}
        except (requests.RequestException,ValueError,KeyError,RuntimeError) as exc:
            return {'ok':False,'tool':tool,'elapsed_seconds':round(time.monotonic()-start,3),'error':str(exc)}
