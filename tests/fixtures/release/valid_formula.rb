class Autobrain < Formula
  include Language::Python::Virtualenv

  resource "pure-python" do
    url "https://files.pythonhosted.org/packages/pure_python-1.0-py3-none-any.whl"
    sha256 "fixture"
  end

  resource "legacy-compatible" do
    url "https://files.pythonhosted.org/packages/legacy_compatible-2.0-py2.py3-none-any.whl"
    sha256 "fixture"
  end

  resource "source-tarball" do
    url "https://files.pythonhosted.org/packages/source_tarball-3.0.tar.gz"
    sha256 "fixture"
  end

  resource "source-zip" do
    url "https://files.pythonhosted.org/packages/source_zip-4.0.zip"
    sha256 "fixture"
  end

  def install
    virtualenv_install_with_resources
  end
end
