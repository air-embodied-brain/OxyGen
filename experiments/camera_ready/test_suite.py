"""CPU-only regression tests for coverage, isolation and portable commands."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run
from validate import validate

class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.config=run.read(run.BUNDLE/'config.example.json')
    def test_complete_manifest(self):
        tasks=run.tasks_for(self.config)
        self.assertEqual(len(tasks),18)
        self.assertEqual(len({x['id'] for x in tasks}),18)
        self.assertEqual({t['label'] for t in tasks if t['kind']=='qwen35'}, {'0.8B-TP1','2B-TP1','4B-TP1','9B-TP1'})
        self.assertEqual(sum(t['kind']=='longrun' for t in tasks),4)
        self.config['include_tp2']=True
        self.assertEqual(len(run.tasks_for(self.config)),24)
    def test_commands_portable_and_no_shell(self):
        self.config['output']='/tmp/a space/$(not-a-command)'
        for task in run.tasks_for(self.config):
            cmd=run.command(task,self.config)
            self.assertIsInstance(cmd,list)
            self.assertFalse(any('/mnt/lixiangyu' in x for x in cmd))
            self.assertFalse(any('{' in x or '}' in x for x in cmd))
            self.assertTrue(any('/tmp/a space/$(not-a-command)' in x for x in cmd))
    def test_empty_arrival_pattern_preserved(self):
        for random_init in (False, True):
            self.config['random_init'] = random_init
            task = next(t for t in run.tasks_for(self.config) if t['id'] == 'longrun_jax_fluctuating')
            cmd = run.command(task, self.config)
            self.assertEqual(cmd[cmd.index('--arrival-pattern') + 1], '')
            self.assertEqual('--random-init' in cmd, random_init)
    def test_random_metadata_changes_affect_resume_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'metadata'; p.mkdir()
            (p / 'config.json').write_text('{}')
            self.config['checkpoints'] = {'model': str(p)}
            before = run.assets(self.config)
            (p / 'config.json').write_text('{"hidden_size": 128}')
            self.assertNotEqual(before, run.assets(self.config))
    def test_physical_device_not_reset_to_zero(self):
        self.config['devices']=['3']
        for t in run.tasks_for(self.config):
            if t['kind'] in ['small_n','pi_reference']:
                cmd=run.command(t,self.config)
                self.assertEqual(cmd[cmd.index('--gpu')+1],'3')
    def test_longrun_measured_frames(self):
        for t in run.tasks_for(self.config):
            if t['kind']!='longrun':continue
            a=t['args'];total=int(a[a.index('--frames')+1]);warm=int(a[a.index('--warmup-frames')+1])
            self.assertEqual(total-warm,3000)
    def test_source_runtime_isolation(self):
        task=run.tasks_for(self.config)[0]
        env=run.environment(task,self.config)
        self.assertIn('vendor/oxygen/src',env['PYTHONPATH'])
        self.assertNotIn(str(run.REPO/'src')+':',env['PYTHONPATH'])
        self.assertEqual(env['HF_HUB_OFFLINE'],'1')
    def test_robot_environment_does_not_leak(self):
        polluted = {key:'/unrelated/robot' for key in
                    ['LD_LIBRARY_PATH','LD_PRELOAD','PYTHONHOME','PYTHONPATH',
                     'CONDA_PREFIX','ROS_DISTRO','CUDA_HOME','VIRTUAL_ENV','RANK']}
        with patch.dict(run.os.environ, polluted):
            env=run.environment(run.tasks_for(self.config)[0],self.config)
        for key in polluted:
            if key != 'PYTHONPATH':self.assertNotIn(key,env)
        self.assertNotIn('/unrelated/robot',env['PYTHONPATH'])
        self.assertEqual(env['PYTHONNOUSERSITE'],'1')
    def test_qwen_implementation_checks_ignore_cross_batch_argmax_but_reject_corruption(self):
        from validate import module
        check = module('analyze_qwen35_scaling_tp').validate_cache_implementation
        audit = dict(input_tokens_exact=True, next_tokens_exact=False,
            prestep_state_max_abs_error={'key':0.0},
            mixed_age_vs_same_age_batch_max_abs_error={'key':0.0},
            mixed_age_vs_same_age_rollout_state_max_abs_error={'key':0.0},
            mixed_age_vs_same_age_rollout_tokens_exact=True,
            slot_recycle_isolation={'recycled_row_vs_fresh_prefix_max_abs_error':{'key':0.0},
                                   'untouched_rows_max_abs_error':{'key':0.0}},
            ages=[0,1], after_ages=[1,2], rollout_steps=30, rollout_after_ages=[30,31])
        check(audit)
        for field in ['prestep_state_max_abs_error','mixed_age_vs_same_age_batch_max_abs_error',
                      'mixed_age_vs_same_age_rollout_state_max_abs_error']:
            for value in (0.5,float('nan'),float('inf')):
                bad=copy.deepcopy(audit);bad[field]['key']=value
                with self.assertRaises(AssertionError):check(bad)
        bad=copy.deepcopy(audit);bad['slot_recycle_isolation']['untouched_rows_max_abs_error']['key']=1
        with self.assertRaises(AssertionError):check(bad)
        bad=copy.deepcopy(audit);bad['after_ages']=[0,2]
        with self.assertRaises(AssertionError):check(bad)

    def test_missing_point_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=next(t for t in run.tasks_for(self.config) if t['kind']=='small_n')
            p=Path(tmp)/t['output'];p.parent.mkdir(parents=True);p.write_text('{"results": []}')
            with self.assertRaisesRegex(ValueError,'missing short-N'):validate(t,tmp)
    def test_missing_repeats_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=next(t for t in run.tasks_for(self.config) if t['kind']=='xiaomi')
            d={'common_prefix_mode':'libero_common','optimization':{'attention':'flash_attention_2','torch_compile':False,'cuda_graph':False},'results':[]}
            p=Path(tmp)/t['output'];p.parent.mkdir(parents=True);p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'incomplete Xiaomi'):validate(t,tmp)
    def test_failed_xiaomi_attempt_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            task=next(t for t in run.tasks_for(self.config) if t['kind']=='xiaomi')
            p=Path(tmp)/task['output'];p.parent.mkdir(parents=True)
            p.with_suffix('.jsonl').write_text('partial')
            other=p.parent/'correctness.json';other.write_text('keep')
            run.archive_attempt(task,tmp,'attempt2')
            self.assertFalse(p.with_suffix('.jsonl').exists())
            self.assertEqual(other.read_text(),'keep')
            self.assertEqual((Path(tmp)/'attempts/xiaomi/attempt2/xiaomi/raw.jsonl').read_text(),'partial')
    def test_assets_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.config['random_init']=False
            p=Path(tmp)/'weights';p.mkdir();(p/'config.json').write_text('{}')
            self.config['checkpoints']={'tiny':str(p)};self.config['assets_lock']=str(Path(tmp)/'lock.json')
            run.assets(self.config,create=True);run.assets(self.config)
            (p/'config.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'differ'):run.assets(self.config)

if __name__=='__main__':unittest.main()
