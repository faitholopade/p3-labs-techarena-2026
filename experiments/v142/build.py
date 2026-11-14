"""Create and verify the submission-only V142 CSV/ZIP and production replay."""
import hashlib,json,sys,zipfile,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v142';DEL=ROOT.parent/'deliverables_v142'
spec=importlib.util.spec_from_file_location('v142_replay',ROOT/'experiments/v142/replay.py');replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def atomic_npz(path,**arrays):
    temp=path.with_suffix('.writing.npz')
    np.savez_compressed(temp,**arrays)
    with zipfile.ZipFile(temp) as archive:assert archive.testzip() is None
    temp.replace(path)
def main():
    confirmation=json.loads((OUT/'confirmation.json').read_text());assert confirmation['date_confirmation_passed'];selection=json.loads((OUT/'selection.json').read_text());assert selection['variant']=='shared' and selection['strength']==.5
    original=ROOT.parent/'deliverables_v136/P3_Labs_submission_v136.csv';assert sha(original)=='380fdb8894ab3b5abdcc31de270b5a704dfba6adc7900a53e90aa0a4b3d9a727';df=pd.read_csv(original,dtype={'rowid':str});base=df['predicted_probability'].to_numpy();data=np.load(ROOT/'runtime/v11/data/evaluation.npz');ids=np.load(ROOT/'runtime/v11/data/evaluation_rowids.npy').astype(str);assert np.array_equal(df['rowid'].to_numpy(),ids);model=dict(np.load(ROOT/'runtime/v141/cohort3/shared.npz'));inputs=dict(items=data['items'],context=data['context'],rowids=ids);assert int(np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'].max())<int(data['timestamp'].min());atomic_npz(OUT/'production_model.npz',**model);atomic_npz(OUT/'production_inputs.npz',**inputs);atomic_npz(OUT/'production_baseline.npz',probabilities=base)
    prediction,delta=replay.predict(model,inputs,base);reference_delta=np.load(ROOT/'runtime/v141/cohort3/predictions.npz')['shared'];assert np.max(abs(delta-reference_delta))<1e-12
    b=base.reshape(-1,10);independent=(softmax(np.log(b)+.5*reference_delta,axis=1)*b.sum(1,keepdims=True)).ravel();independent_error=float(abs(independent-prediction).max());assert independent_error<1e-14;assert len(prediction)==529390 and np.isfinite(prediction).all() and prediction.min()>0 and prediction.max()<1 and len(set(ids))==len(ids);mass_error=float(abs(prediction.reshape(-1,10).sum(1)-b.sum(1)).max());assert mass_error<1e-12
    # Candidate permutation must only permute item probabilities and deltas.
    small={k:v[:128] if k!='rowids' else v[:1280] for k,v in inputs.items()};permutation=np.random.default_rng(142026).permutation(10);small['items']=small['items'][:,permutation];permuted,_=replay.predict(model,small,b[:128,permutation].ravel());permutation_error=float(abs(permuted.reshape(-1,10)-prediction.reshape(-1,10)[:128,permutation]).max());assert permutation_error<1e-8
    DEL.mkdir(exist_ok=True);csv=DEL/'P3_Labs_submission_v142.csv';pd.DataFrame({'rowid':ids,'predicted_probability':prediction}).to_csv(csv,index=False,float_format='%.10g',lineterminator='\n');read=pd.read_csv(csv,dtype={'rowid':str});assert list(read.columns)==['rowid','predicted_probability'] and np.array_equal(read['rowid'].to_numpy(),ids);rounding=float(abs(read['predicted_probability'].to_numpy()-prediction).max());assert rounding<=5.01e-11;archive=DEL/(csv.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.write(csv,csv.name)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==[csv.name] and z.testzip() is None
        with z.open(csv.name) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha(csv)
    verification=dict(version='V142',classification='Exploratory competition submission',baseline_version='V136',baseline_reported_public_auc=.832744,baseline_csv_sha256=sha(original),csv_file=csv.name,csv_sha256=sha(csv),csv_bytes=csv.stat().st_size,zip_file=archive.name,zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,rows=len(prediction),unique_ids=len(set(ids)),id_order_exact=True,finite_probabilities=True,minimum_probability=float(prediction.min()),maximum_probability=float(prediction.max()),mean_absolute_probability_change=float(abs(prediction-base).mean()),maximum_absolute_probability_change=float(abs(prediction-base).max()),preserved_event_mass_max_error=mass_error,independent_inference_max_error=independent_error,csv_rounding_max_error=rounding,candidate_permutation_check=True,candidate_permutation_max_error=permutation_error,candidate_permutation_tolerance=1e-8,zip_contains_one_csv=True,zip_crc_verified=True,production_model_sha256=sha(OUT/'production_model.npz'),production_inputs_sha256=sha(OUT/'production_inputs.npz'),maximum_training_timestamp=int(np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'].max()),minimum_evaluation_timestamp=int(data['timestamp'].min()),public_score=None,portal_uploads=0)
    write(OUT/'submission_verification.json',verification);print(json.dumps(verification),flush=True)
if __name__=='__main__':main()
