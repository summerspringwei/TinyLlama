a = 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-install-sanv1imm/xformers_9fa64e89eb7341259c39a091bfc4df42/setup.py'"'"'; __file__='"'"'/tmp/pip-install-sanv1imm/xformers_9fa64e89eb7341259c39a091bfc4df42/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))'
com = a.split(';')
for c in com:
    print(c.strip())