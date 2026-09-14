import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook} from '@oai/artifact-tool';

const out=process.argv[2];
if(!out) throw new Error('Output directory required');
const tables=JSON.parse(await fs.readFile(path.join(out,'prepared_tables.json'),'utf8'));
const escapeCSV=x=>{
  const s=x===null||x===undefined?'':String(x);
  return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;
};
const verification=[];
for(const [name,table] of Object.entries(tables)){
  const w=Workbook.create();
  const s=w.worksheets.add('Data');
  const matrix=[table.columns,...table.rows];
  const cols=table.columns.length;
  for(let start=0;start<matrix.length;start+=1500){
    const block=matrix.slice(start,start+1500);
    s.getRangeByIndexes(start,0,block.length,cols).values=block;
  }
  w.recalculate();
  const got=s.getRangeByIndexes(0,0,matrix.length,cols).values;
  for(let r=0;r<matrix.length;r++)for(let c=0;c<cols;c++){
    const expected=matrix[r][c],actual=got[r]?.[c];
    if((actual??'')!==(expected??''))throw new Error(`Cell changed: ${name} ${r},${c}`);
  }
  if(name==='hotpotqa_all_methods.csv'){
    const area=s.getRangeByIndexes(0,0,matrix.length,7);
    area.format.font={name:'Arial',size:10};
    for(const [c,width] of [[0,12],[1,21],[2,17],[3,38],[4,9],[5,19],[6,21]])
      s.getRangeByIndexes(0,c,matrix.length,1).format.columnWidth=width;
    s.getRangeByIndexes(0,0,1,7).format.font={bold:true};
    s.getRangeByIndexes(1,6,matrix.length-1,1).setNumberFormat('0.00%');
    console.log((await w.inspect({kind:'table',range:'Data!A1:G15',include:'values',tableMaxRows:15,tableMaxCols:7,maxChars:3000})).ndjson);
    const preview=await w.render({sheetName:'Data',range:'A1:G15',scale:1,format:'png'});
    await fs.writeFile(path.join(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?=[A-Za-z]:)/,'')),'summary-preview.png'),new Uint8Array(await preview.arrayBuffer()));
  }
  // Public CSV-export discovery returned no supported API. Serialize the validated
  // artifact-tool cell values to the requested plain CSV, preserving its schema.
  const csv=got.map(row=>row.map(escapeCSV).join(',')).join('\r\n')+'\r\n';
  await fs.writeFile(path.join(out,name),csv,'utf8');
  verification.push({file:name,rows:table.rows.length,columns:cols,all_cells_preserved:true});
  console.log(JSON.stringify({written:name,rows:table.rows.length}));
}
await fs.writeFile(path.join(out,'csv_export_verification.json'),JSON.stringify({author:'@oai/artifact-tool',format:'csv',files:verification},null,2)+'\n');
