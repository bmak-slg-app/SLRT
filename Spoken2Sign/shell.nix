{
  pkgs ? import <nixpkgs> { },
  lib,
  python ? pkgs.python310Full,
}:

with pkgs;

mkShell {
  packages =
    [
      python
    ]
    ++ (with pkgs; [
      blender
      ffmpeg
      uv
      valkey
    ]);
  env =
    {
      # Prevent uv from managing Python downloads
      UV_PYTHON_DOWNLOADS = "never";
      # Force uv to use nixpkgs Python interpreter
      UV_PYTHON = python.interpreter;
    }
    // lib.optionalAttrs pkgs.stdenv.isLinux {
      # Python libraries often load native shared objects using dlopen(3).
      # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
      LD_LIBRARY_PATH = lib.makeLibraryPath (
        pkgs.pythonManylinuxPackages.manylinux1
        ++ [
          # stdenv.cc.cc.lib
          # glib
          # zlib
          xorg.libX11
          xorg.libXrender
          xorg.libXxf86vm
          xorg.libXfixes
          xorg.libXi
          xorg.libSM
          xorg.libICE
          libxkbcommon
        ]
      );
      # NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";
      SSL_CERT_FILE = "/etc/ssl/certs/ca-bundle.crt";
    };
  shellHook = ''
    unset PYTHONPATH
  '';
}
