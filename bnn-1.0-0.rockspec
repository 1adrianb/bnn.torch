package = "bnn"
version = "1.0-0"

source = {
   url = "git://github.com/1adrianb/bnn.torch",
   tag = "master"
}

description = {
   summary = "torch binary c/nn repository",
   detailed = [[
torch binary nn routines
   ]],
   homepage = "https://github.com/1adrianb/bnn.torch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
   ]],
   install_command = "cd build && $(MAKE) install"
}