import glob,os,re,collections
R='<ORIGINAL_ROOT>/workspace/experiments/verification/dft_results_merged'
c=collections.Counter(); who=collections.defaultdict(list)
for d in sorted(glob.glob(R+"/*")):
    f=d+"/OUTCAR"
    if not os.path.isfile(f): continue
    l=open(f,errors="replace").readline()
    v=re.search(r"vasp\.(\d+\.\d+\.\d+)",l).group(1)
    c[v]+=1; who[v].append(os.path.basename(d)+"<-"+os.path.basename(os.path.dirname(os.path.realpath(d))))
print(dict(c),sum(c.values()))
for v in who:
    print(v,len(who[v])); print("  ", who[v])
# pair-level mixing: pair id = gb_<model>_<t>_{hi,lo} and mu_<model>_<t>_{hi,lo} (proposed uses mu_proposed_common)
pairs=collections.defaultdict(set)
for v in who:
    for s in who[v]:
        n=s.split("<-")[0]
        m=re.match(r"(?:gb|mu)_(proposed|baseline)_(t\d)",n)
        if m: pairs[m.group(1)+"_"+m.group(2)].add(v)
        elif n=="mu_proposed_common": 
            for t in range(7): pairs["proposed_t%d"%t].add(v)
print({k:sorted(v) for k,v in sorted(pairs.items())})
